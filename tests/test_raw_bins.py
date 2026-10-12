"""Exercise the binary reader and viewer against actual raw layout."""
import json
from pathlib import Path

import numpy as np
import pytest

from optcar.datagen.raw_bins import action_window, bin_frame, inspect_bin, scan_bins
from optcar.gui.server import create_app


def recording(path, steps=20, columns=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.zeros((steps, columns), dtype='<f4')
    rows[:, 0] = np.arange(steps) * .1
    rows[:, 3] = 1
    rows[:, 7] = 5
    rows[:, 13] = 2
    rows[:, 14] = .3
    rows.tofile(path)
    return rows


def test_nested_metadata_and_exact_last_frame(tmp_path):
    file = tmp_path / 'data/source/terrain/session/run.bin'
    rows = recording(file, steps=7003)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    (metadata / 'schema.json').write_text(json.dumps({'features': 15}))
    (metadata / 'conversion.json').write_text(json.dumps({'config': {'dt': .04}, 'bins': {str(file.relative_to(tmp_path)): {'source': '/recording/run'}}}))
    (metadata / 'dataset_config.py').write_text('raise RuntimeError("must never execute")\nclass DatasetConfig:\n    num_envs: int = 1\n    control_dt: float = 0.04\n')
    scan = scan_bins(str(tmp_path))
    assert len(scan['bins']) == 1
    assert scan['bins'][0]['dt'] == .04
    data = inspect_bin(str(file))
    assert data['source'] == '/recording/run'
    assert len(data['rows']) == 6000
    assert data['indices'][-1] == 7002
    assert data['stats'][0]['max'] == float(rows[-1, 0])
    assert bin_frame(str(file), 7002)['row'] == rows[-1].tolist()
    assert bin_frame(str(file), 7002)['time'] == 7002 * .04


@pytest.mark.parametrize('width', [15, 17])
def test_headerless_layout_and_nonfinite_json(tmp_path, width):
    file = tmp_path / 'raw.bin'
    rows = recording(file, steps=255, columns=width)  # divisible by both widths
    rows[3, 1] = np.nan
    rows[5, 9] = np.inf
    rows.tofile(file)
    data = inspect_bin(str(file))
    assert data['columns'] == width
    assert data['rows'][3][1] is None
    assert data['stats'][9]['missing'] == 1
    json.dumps(data, allow_nan=False)


def test_multi_environment_order_and_invalid_files(tmp_path):
    file = tmp_path / 'run.bin'
    rows = recording(file, steps=20)
    rows[10:, 0] += 100
    rows.tofile(file)
    data = inspect_bin(str(file), columns=15, num_envs=2, env_index=1)
    assert data['samples'] == 10
    assert data['frame'] == rows[10].tolist()
    assert bin_frame(str(file), 9, num_envs=2, env_index=1)['row'] == rows[-1].tolist()
    with pytest.raises(ValueError):
        bin_frame(str(file), 10, num_envs=2)
    with pytest.raises(ValueError):
        inspect_bin(str(file), num_envs=0)
    with pytest.raises(ValueError):
        inspect_bin(str(file), dt=-.01)
    bad = tmp_path / 'partial.bin'
    bad.write_bytes(b'123')
    scan = scan_bins(str(tmp_path))
    assert len(scan['bins']) == 2
    assert scan['bins'][0]['error']


@pytest.mark.parametrize('width', [15, 17])
def test_packed_synthetic_batch_metadata_routes(tmp_path, width):
    file = tmp_path / 'raw/batch.bin'
    first = recording(file, steps=501, columns=width)
    second = first.copy()
    second[:, 1] = np.arange(501) * .2
    second[:, 7] = 3
    second[:, 13] = 4
    np.stack([first, second]).tofile(file)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    (metadata / 'dataset_config.py').write_text(
        f'class DatasetConfig:\n    num_envs = 2\n    control_dt = .02\n    feature_dim = {width}\n')
    client = create_app(tmp_path / 'datasets').test_client()
    scan = client.post('/api/raw-bins/scan', json={'path': str(tmp_path)}).json
    assert scan['bins'][0]['num_envs'] == 2
    assert scan['bins'][0]['samples'] == 501 and scan['bins'][0]['duration_s'] == 10
    request = {'path': str(file), 'env_index': 1}
    view = client.post('/api/raw-bins/open', json=request).json
    assert view['env_index'] == 1 and view['frame'] == second[0].tolist()
    frame = client.post('/api/raw-bins/frame', json={**request, 'index': 500}).json
    assert frame['row'] == second[-1].tolist() and frame['forward_velocity'] == 3
    actions = client.post('/api/raw-bins/actions', json={**request, 'index': 250, 'before_s': .04, 'after_s': .02}).json
    assert actions['commands'] == second[248:252, 13:].tolist()
    assert actions['forward_velocity'] == [3.] * 4
    assert client.post('/api/raw-bins/open', json={**request, 'env_index': 2}).status_code == 400


def test_viewer_routes_and_bad_metadata_width(tmp_path):
    file = tmp_path / 'data/run.bin'
    rows = recording(file)
    client = create_app(dataset_root=tmp_path / 'datasets').test_client()
    scan = client.post('/api/raw-bins/scan', json={'path': str(tmp_path)}).get_json()
    assert scan['bins'][0]['samples'] == 20
    data = client.post('/api/raw-bins/open', json={'path': str(file)}).get_json()
    frame = client.post('/api/raw-bins/frame', json={'path': str(file), 'index': 19}).get_json()
    assert data['rows'][0] == rows[0].tolist()
    assert frame['row'] == rows[-1].tolist()
    assert client.post('/api/raw-bins/frame', json={'path': str(file), 'index': 20}).status_code == 400
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    (metadata / 'schema.json').write_text('{"features":17}')
    assert client.post('/api/raw-bins/open', json={'path': str(file)}).status_code == 400


def test_forward_velocity_uses_full_quaternion(tmp_path):
    from scipy.spatial.transform import Rotation
    file = tmp_path / 'run.bin'
    rows = recording(file, steps=3)
    rotation = Rotation.from_euler('xyz', [[0, 0, 90], [0, 45, 0], [20, 30, -70]], degrees=True)
    body = np.array([[4., 0., 0.], [-3., 2., 1.], [1.5, -4., 2.]])
    quat = rotation.as_quat()
    rows[:, 3:7] = quat[:, [3, 0, 1, 2]]
    rows[:, 7:10] = rotation.apply(body)
    rows.tofile(file)
    view = inspect_bin(str(file))
    np.testing.assert_allclose(view['forward_velocity'], body[:, 0], atol=1e-6)
    assert view['forward_source'] == 'Derived from world velocity'
    assert bin_frame(str(file), 2)['forward_velocity'] == pytest.approx(1.5, abs=1e-6)
    assert view['forward_stats']['min'] == pytest.approx(-3., abs=1e-6)


def test_invalid_auxiliary_falls_back_with_warning(tmp_path):
    file = tmp_path / 'data/run.bin'
    recording(file)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    record = {'aux': {'frame': 'body', 'path': 'aux/run.bin', 'topic': '/twist'}}
    (metadata / 'conversion.json').write_text(json.dumps({'bins': {'data/run.bin': record}}))
    view = inspect_bin(str(file))
    assert view['forward_velocity'][0] == 5
    assert 'missing or has a different row count' in view['aux_warning']


def test_action_window_tracks_current_sample_and_timestep(tmp_path):
    file = tmp_path / 'run.bin'
    rows = recording(file, steps=10003, columns=17)
    rows[:, 13] = np.arange(len(rows))
    rows.tofile(file)
    # A narrow window must read every sample, even on a sampled overview.
    window = action_window(str(file), 5001, .12, .08, dt=.04, columns=17)
    assert window['indices'] == list(range(4998, 5004))
    assert window['current'] == 5001
    assert window['commands'] == rows[4998:5004, 13:].tolist()
    assert window['forward_velocity'] == [5.] * 6
    assert action_window(str(file), 0, 5, .08, dt=.04, columns=17)['indices'] == [0, 1, 2]
    assert action_window(str(file), 10002, .08, 5, dt=.04, columns=17)['indices'] == [10000, 10001, 10002]
    wide = action_window(str(file), 5001, 1000, 1000, columns=17)
    assert len(wide['indices']) <= 6001
    assert 5001 in wide['indices']
    for before, after in [(0, 0), (-1, 2), (float('nan'), 2), (1, float('inf'))]:
        with pytest.raises(ValueError):
            action_window(str(file), 5001, before, after, columns=17)


def test_action_window_route_uses_aligned_auxiliary(tmp_path):
    file = tmp_path / 'data/run.bin'
    recording(file)
    aux = tmp_path / 'aux/run.bin'
    aux.parent.mkdir()
    twist = np.zeros((20, 6), dtype='<f4')
    twist[:, 0] = np.arange(20) * .25
    twist.tofile(aux)
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    record = {'aux': {'frame': 'body', 'path': 'aux/run.bin', 'topic': '/twist'}}
    (metadata / 'conversion.json').write_text(json.dumps({'bins': {'data/run.bin': record}}))
    client = create_app(dataset_root=tmp_path / 'datasets').test_client()
    payload = {'path': str(file), 'index': 10, 'before_s': .04, 'after_s': .02, 'dt': .02}
    response = client.post('/api/raw-bins/actions', json=payload)
    assert response.status_code == 200
    window = response.get_json()
    assert window['indices'] == [8, 9, 10, 11]
    assert window['forward_velocity'] == twist[8:12, 0].tolist()
    assert client.post('/api/raw-bins/actions', json={**payload, 'index': 20}).status_code == 400
    assert client.post('/api/raw-bins/actions', json={**payload, 'before_s': 0, 'after_s': 0}).status_code == 400
