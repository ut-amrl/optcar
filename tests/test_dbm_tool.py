"""Check the copied manual workbench on raw bins before training-window processing."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from optcar.datagen.raw_bins import body_frame_window
from optcar.gui.server import create_app
from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel


@pytest.fixture
def manual_recording(tmp_path):
    dt = .02
    initial = np.zeros(13, dtype=np.float32)
    initial[:3] = [10, -8, .3]
    yaw = .8
    initial[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    initial[7:10] = [np.cos(yaw), np.sin(yaw), 0]
    commands = np.column_stack([np.linspace(1, 1.6, 60), np.full(60, .1)]).astype(np.float32)
    model = DynamicBicycleModel()
    with torch.inference_mode():
        future, _ = model.rollout(initial[None], commands[None], dt)
    states = np.vstack([initial, future[0].numpy()])
    rows = np.column_stack([states, np.vstack([commands, commands[-1:]])]).astype('<f4')
    file = tmp_path / 'data/source/mulch/run.bin'
    file.parent.mkdir(parents=True)
    rows.tofile(file)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    (metadata / 'schema.json').write_text(json.dumps({'features': 15}))
    (metadata / 'conversion.json').write_text(json.dumps({'config': {'dt': dt}}))
    client = create_app(dataset_root=tmp_path / 'datasets').test_client()
    loaded = client.post('/dbm/api/load', json={'bin_path': str(file)})
    assert loaded.status_code == 200
    return client, loaded.get_json(), file, rows, dt


def test_raw_bin_inspection_and_body_rollout(manual_recording):
    client, loaded, file, rows, dt = manual_recording
    scan = client.post('/dbm/api/inspect', json={'dataset_path': str(file.parents[3])}).get_json()
    assert scan['bins'][0]['path'] == str(file)
    assert loaded['n_cols'] == 15
    assert loaded['duration'] == pytest.approx((len(rows) - 1) * dt)
    payload = {'session_id': loaded['session_id'], 'frame_idx': 0, 'horizon': 60, 'cmd_shift_s': 0}
    measured = client.post('/dbm/api/frame', json=payload).get_json()
    prediction = client.post('/dbm/api/rollout', json={**payload, 'params': asdict(DBMParams())})
    assert prediction.status_code == 200, prediction.get_json()
    predicted = prediction.get_json()
    assert len(measured['t']) == len(predicted['bx']) == 61
    assert measured['bx'][0] == measured['by'][0] == 0
    np.testing.assert_allclose(predicted['bx'], measured['bx'], atol=2e-5)
    np.testing.assert_allclose(predicted['by'], measured['by'], atol=2e-5)
    changed = client.post('/dbm/api/rollout', json={**payload, 'params': {**asdict(DBMParams()), 'C_tire_f': 0, 'C_tire_r': 0}}).get_json()
    assert not np.allclose(predicted['by'], changed['by'])


def test_command_delay_and_recording_end(manual_recording):
    client, loaded, file, rows, dt = manual_recording
    payload = {'session_id': loaded['session_id'], 'frame_idx': 4, 'horizon': 5, 'cmd_shift_s': 2 * dt}
    frame = client.post('/dbm/api/frame', json=payload).get_json()
    np.testing.assert_allclose(frame['cmd_v'], rows[2:8, 13])
    end = client.post('/dbm/api/frame', json={**payload, 'frame_idx': len(rows) - 3, 'horizon': 100}).get_json()
    assert len(end['bx']) == 3
    pred = client.post('/dbm/api/rollout', json={**payload, 'frame_idx': len(rows) - 3, 'horizon': 100, 'params': asdict(DBMParams())}).get_json()
    assert len(pred['bx']) == 3
    assert client.post('/dbm/api/rollout', json={**payload, 'frame_idx': 0, 'params': asdict(DBMParams())}).status_code == 400
    for patch in [{'horizon': 0}, {'horizon': 10001}, {'frame_idx': -1}, {'frame_idx': .5}]:
        assert client.post('/dbm/api/frame', json={**payload, **patch}).status_code == 400
    assert client.post('/dbm/api/rollout', json={**payload, 'params': {**asdict(DBMParams()), 'mass': 0}}).status_code == 400


def test_unshifted_commands_for_17_column_bins(manual_recording):
    client, loaded, file, rows, dt = manual_recording
    extended = np.column_stack([rows, np.column_stack([np.full(len(rows), 4), np.full(len(rows), -.2)])]).astype('<f4')
    extended.tofile(file)
    (file.parents[3] / 'metadata/schema.json').write_text(json.dumps({'features': 17}))
    session = client.post('/dbm/api/load', json={'bin_path': str(file)}).get_json()
    frame = client.post('/dbm/api/frame', json={'session_id': session['session_id'], 'frame_idx': 4, 'horizon': 5}).get_json()
    assert frame['cmd_v'] == [4.] * 6
    np.testing.assert_allclose(frame['cmd_s'], [-.2] * 6)


def test_segments_save_load_and_restore(manual_recording):
    client, loaded, file, rows, dt = manual_recording
    saved = client.post('/dbm/api/save_segment', json={'session_id': loaded['session_id'], 'start_idx': 4, 'end_idx': 15}).get_json()
    assert saved['start_idx'] == 4
    path = saved['yaml_path']
    segment = client.post('/dbm/api/load_segment', json={'yaml_path': path}).get_json()
    assert segment['end_idx'] == 15
    reread = client.post('/dbm/api/load', json={'bin_path': str(file)}).get_json()
    assert reread['saved_segment']['start_idx'] == 4


@pytest.mark.parametrize('name, params', [
    ('CircleTrajectory', {'radius': 2, 'num_points': 100, 'center_xy': [1, 2]}),
    ('Figure8Trajectory', {'num_points': 100}),
    ('DubinsTrajectory', {'goal_pose': [3, 2, .5], 'cand_r_start': [1.], 'cand_r_end': [1.], 'n_jobs': 1}),
])
def test_reference_trajectories(manual_recording, name, params):
    client, *_ = manual_recording
    response = client.post('/dbm/api/reference_trajectory', json={'trajectory_name': name, 'params': params, 'max_points': 80})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert len(data['x']) == len(data['y']) > 2
    assert np.isfinite(data['x']).all()


def test_preset_round_trip_and_local_assets(manual_recording):
    client, _, file, *_ = manual_recording
    params = {**asdict(DBMParams()), 'C_tire_f': 32.}
    location = {'folder': str(file.parent / 'my configs'), 'filename': 'mulch.yml'}
    response = client.post('/api/preset', json={**location, 'params': params})
    assert response.status_code == 200
    assert response.json['path'] == str(file.parent / 'my configs' / 'mulch.yml')
    assert client.get('/api/preset', query_string=location).get_json()['params'] == params
    # Missing choices cannot write to an implicit terrain/default location.
    for patch in [{}, {'folder': location['folder']}, {'filename': 'mulch.yml'},
                  {**location, 'filename': '../mulch.yml'}, {**location, 'folder': 'relative'}]:
        assert client.post('/api/preset', json={**patch, 'params': params}).status_code == 400
    assert client.get('/api/preset', query_string={'folder': location['folder']}).status_code == 400
    html = client.get('/dbm/?path=/my/bin.bin').data.decode()
    assert 'id="preset_folder"' in html and 'id="preset_filename"' in html
    assert 'id="preset_terrain"' not in html
    assert 'value="/my/bin.bin"' in html
    assert 'cdn.plot.ly' not in html
    assert client.get('/dbm/static/plotly-2.35.2.min.js').status_code == 200


def test_custom_reference_from_waypoint_file(manual_recording):
    client, _, file, *_ = manual_recording
    waypoints = file.parent / 'reference.txt'
    waypoints.write_text('0 0\n2 0\n2 2\n0 2\n')
    response = client.post('/dbm/api/reference_trajectory', json={
        'trajectory_name': 'CustomClosedLoopTrajectory',
        'params': {'txt_file': str(waypoints), 'num_resample_points': 100}})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert len(data['x']) > 2
    assert np.isfinite(data['x']).all()


def test_nonfinite_diagnostic_values_are_json_gaps(manual_recording):
    client, _, file, rows, _ = manual_recording
    rows[5, 7] = np.inf
    rows[6, 12] = np.nan
    rows.tofile(file)
    loaded = client.post('/dbm/api/load', json={'bin_path': str(file)}).get_json()
    response = client.post('/dbm/api/frame', json={'session_id': loaded['session_id'], 'frame_idx': 0, 'horizon': 10})
    assert response.status_code == 200
    assert response.get_json()['v_long'][5] is None
    assert response.get_json()['yaw_rate'][6] is None
    assert b'Infinity' not in response.data and b'NaN' not in response.data
