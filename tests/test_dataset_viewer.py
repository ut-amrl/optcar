"""Processed viewer fixtures exercise the saved FKD columns and shared loader."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch
import yaml

from optcar.dataset import DataConfig, DatasetWriter, FKDDataset
from optcar.gui.dataset_viewer import (
    DatasetViewer, load_viewer_config, read_dataset_config, save_viewer_config,
)
from optcar.gui.server import create_app


def config_fixture(path, **changes):
    values = {'history_steps': 3, 'future_steps': 2, 'dt': .04,
              'stride': 2, 'shard_windows': 2, **changes}
    path.write_text(yaml.safe_dump({'data': values}))
    return path


def packed_fixture(count=2, offset=0):
    rows = np.zeros((count, 5, 42), dtype=np.float32)
    times = np.arange(5, dtype=np.float32)
    for window in range(count):
        rows[window, :, 0] = offset + window * 20 + times
        rows[window, :, 1] = times * .2
        rows[window, :, 3] = np.cos(np.pi / 4)
        rows[window, :, 6] = np.sin(np.pi / 4)
        rows[window, :, 7:10] = [4, 2, .5]
        rows[window, :, 13] = (times - 2) * .25
        rows[window, :, 14] = (times - 2) * -.3
        rows[window, :, 16] = np.cos(np.pi / 8)
        rows[window, :, 19] = -np.sin(np.pi / 8)
        rows[window, :, 20:26] = [3, 2, 1, .1, .2, .3]
        rows[window, :, 26:29] = [.1, .2, .3]
        rows[window, :, 29] = np.cos(np.pi / 12)
        rows[window, :, 32] = np.sin(np.pi / 12)
        rows[window, :, 33:39] = [9, 8, 7, .4, .5, .6]
        rows[window, :, 39] = times * .7 - .5
        rows[window, :, 40] = times * -.1 + .2
        rows[window, :, 41] = offset + window + 7
    return rows


def assembled_fixture(root, cfg, flat=False):
    expected = {name: packed_fixture(count, offset) for name, count, offset in
                [('train', 2, 0), ('val', 1, 100), ('test', 1, 200)]}
    with DatasetWriter(root, cfg, 'sim', 'road', flat_processed=flat) as writer:
        for split, rows in expected.items():
            writer.write_window_batches(split + '_input', [torch.from_numpy(rows)],
                split, split + '_session', root.parent / (split + '.bin'), [1, 8, 15])
    return expected


def open_fixture(client, root, config_path, split='train'):
    response = client.post('/api/dataset-viewer/open', json={
        'roots': [str(root)], 'config_path': str(config_path), 'split': split})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def test_config_formats_horizons_and_python_metadata_never_execute(tmp_path):
    path = config_fixture(tmp_path / 'optcar.yaml')
    config = read_dataset_config(path)
    assert asdict(config) == {'history_steps': 3, 'future_steps': 2, 'dt': .04,
                              'stride': 2, 'shard_windows': 2}

    marker = tmp_path / 'executed.txt'
    python_config = tmp_path / 'dataset_config.py'
    python_config.write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n'
        'raise RuntimeError("must never execute")\n'
        'class ProcessedDatasetConfig:\n'
        '    history_horizon: int = 3\n'
        '    future_horizon: int = 3\n'
        '    total_horizon: int = 5\n'
        '    control_dt: float = .04\n'
        '    feature_dim: int = 42\n'
        '    data_dtype: str = "float32"\n')
    config = read_dataset_config(python_config)
    assert (config.history_steps, config.future_steps, config.dt, config.window_steps) == (3, 2, .04, 5)
    assert not marker.exists()

    schema = tmp_path / 'schema.json'
    schema.write_text(json.dumps({'history_horizon': 3, 'future_horizon': 3,
        'total_horizon': 5, 'sim_dt': .01, 'decimation': 4,
        'features': 42, 'dtype': 'float32', 'quaternion_order': 'wxyz'}))
    config = read_dataset_config(schema)
    assert (config.history_steps, config.future_steps, config.dt) == (3, 2, .04)


@pytest.mark.parametrize('changes', [
    {'history_steps': 1}, {'history_steps': 3.5}, {'history_steps': True},
    {'future_steps': 0}, {'future_steps': 2.5}, {'future_steps': True},
    {'dt': 0}, {'dt': float('nan')}, {'dt': True}, {'features': 41},
    {'feature_dim': 15}, {'dtype': 'float64'}, {'data_dtype': 'float16'},
    {'quaternion_order': 'xyzw'}, {'total_horizon': 6}, {'chunk_horizon': 6},
])
def test_config_rejects_incompatible_dimensions_and_layout(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        read_dataset_config(config_fixture(tmp_path / 'bad.yaml', **changes))


def test_missing_or_dynamic_required_config_values_are_not_defaulted(tmp_path):
    path = tmp_path / 'missing.yaml'
    path.write_text('data:\n  dt: 0.04\n')
    with pytest.raises(ValueError):
        read_dataset_config(path)
    path = tmp_path / 'dynamic.py'
    path.write_text('class ProcessedDatasetConfig:\n'
                    '    history_horizon = compute_history()\n'
                    '    future_horizon = 3\n    control_dt = .04\n')
    with pytest.raises(ValueError):
        read_dataset_config(path)


@pytest.mark.parametrize('name, text', [
    ('invalid.py', 'class ProcessedDatasetConfig:\n    history_horizon = (\n'),
    ('invalid.yaml', 'data:\n  history_steps: [3\n'),
])
def test_api_reports_malformed_config_as_a_client_error(tmp_path, name, text):
    root = tmp_path / 'plain'
    root.mkdir()
    packed_fixture(1).tofile(root / 'sample.bin')
    config_path = tmp_path / name
    config_path.write_text(text)
    client = create_app(tmp_path / 'gui').test_client()
    response = client.post('/api/dataset-viewer/open', json={
        'roots': [str(root)], 'config_path': str(config_path), 'split': 'all'})
    assert response.status_code == 400
    assert 'Cannot read dataset config' in response.get_json()['error']
    assert str(config_path) in response.get_json()['error']


def test_api_reads_exact_saved_columns_and_standard_slices(tmp_path, monkeypatch):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'assembled'
    expected = assembled_fixture(root, cfg)
    config_path = config_fixture(tmp_path / 'config.yaml')
    client = create_app(tmp_path / 'gui').test_client()
    calls = []
    original = FKDDataset.get_fkd_sample

    def shared_loader(dataset, index):
        calls.append(index)
        return original(dataset, index)

    monkeypatch.setattr(FKDDataset, 'get_fkd_sample', shared_loader)
    opened = open_fixture(client, root, config_path)
    assert opened['count'] == 2 and opened['files'] == 1
    assert opened['samples'] == 5 and opened['features'] == 42
    assert opened['history_steps'] == 3 and opened['future_steps'] == 2
    assert opened['dt'] == .04 and calls == []
    manifest = json.loads((root / 'manifest.json').read_text())
    shard = root / manifest['processed_shards'][0]['path']
    original_bytes = shard.read_bytes()

    response = client.post('/api/dataset-viewer/sample', json={
        'dataset_id': opened['dataset_id'], 'index': 1})
    assert response.status_code == 200, response.get_json()
    sample = response.get_json()
    row = expected['train'][1]
    assert calls == [1]
    assert sample['index'] == 1 and sample['file_index'] == 0 and sample['window_index'] == 1
    assert sample['dataset_idx'] == 0 and sample['source'] == 'sim' and sample['terrain'] == 'road'
    assert sample['split'] == 'train' and sample['path'] == str(shard)
    assert sample['car_number'] == 8 and sample['current_step'] == 2
    np.testing.assert_allclose(sample['times'], [-.08, -.04, 0, .04, .08])
    for key, section in [('world', slice(0, 13)), ('body', slice(13, 26)),
                         ('delta', slice(26, 39)), ('actions', slice(39, 41))]:
        np.testing.assert_array_equal(sample[key], row[:, section])
    np.testing.assert_array_equal(sample['history'], row[:3, 13:26])
    np.testing.assert_array_equal(sample['future'], row[2:, 13:26])
    np.testing.assert_array_equal(sample['history_actions'], row[:3, 39:41])
    np.testing.assert_array_equal(sample['future_actions'], row[2:, 39:41])
    np.testing.assert_array_equal(sample['delta_history'], row[:3, 26:39])
    delta_future = row[2:, 26:39].copy()
    delta_future[0, :7] = [0, 0, 0, 1, 0, 0, 0]
    np.testing.assert_array_equal(sample['delta_future'], delta_future)
    np.testing.assert_allclose(sample['world_yaw'], np.pi / 2, atol=1e-6)
    np.testing.assert_allclose(sample['body_yaw'], -np.pi / 4, atol=1e-6)
    assert shard.read_bytes() == original_bytes
    repeated = client.post('/api/dataset-viewer/sample', json={
        'dataset_id': opened['dataset_id'], 'index': 1}).get_json()
    assert repeated == sample

    for split, count, files in [('val', 1, 1), ('test', 1, 1), ('all', 4, 3)]:
        result = open_fixture(client, root, config_path, split)
        assert result['count'] == count and result['files'] == files


@pytest.mark.parametrize('config_name', ['dataset_config.py', 'schema.json'])
def test_flat_layout_viewer_uses_saved_metadata_and_valid_folder(tmp_path, config_name):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'processed' / 'sim' / 'road'
    expected = assembled_fixture(root, cfg, flat=True)
    config_path = root / 'metadata' / config_name
    saved_config = read_dataset_config(config_path)
    assert (saved_config.history_steps, saved_config.future_steps, saved_config.dt) == (3, 2, .04)
    assert not (root / 'processed').exists()
    client = create_app(tmp_path / 'gui').test_client()
    opened = open_fixture(client, root / 'valid', config_path, split='val')
    assert opened['count'] == 1 and opened['files'] == 1
    response = client.post('/api/dataset-viewer/sample', json={
        'dataset_id': opened['dataset_id'], 'index': 0})
    assert response.status_code == 200, response.get_json()
    sample = response.get_json()
    assert sample['split'] == 'val'
    assert sample['path'] == str(root / 'valid' / 'processed_0000000000.bin')
    np.testing.assert_array_equal(sample['world'], expected['val'][0, :, :13])
    np.testing.assert_array_equal(sample['body'], expected['val'][0, :, 13:26])
    assert open_fixture(client, root.parent.parent, config_path, split='all')['count'] == 4
    assert client.post('/api/dataset-viewer/open', json={
        'roots': [str(root / 'valid')], 'config_path': str(config_path), 'split': 'train'}).status_code == 400


def test_viewer_combines_legacy_and_flat_paths_without_duplicate_windows(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    old_root = tmp_path / 'old' / 'road' / 'sim'
    new_root = tmp_path / 'new' / 'processed' / 'sim' / 'road'
    old_expected = assembled_fixture(old_root, cfg)
    new_expected = assembled_fixture(new_root, cfg, flat=True)
    viewer = DatasetViewer()
    opened = viewer.open([str(new_root / 'valid'), str(old_root / 'processed'),
                          str(new_root), str(old_root)],
                         str(new_root / 'metadata' / 'dataset_config.py'), 'all')
    assert opened['count'] == 8 and opened['files'] == 6
    first = viewer.sample(opened['dataset_id'], 0)
    assert first['split'] == 'val' and first['path'].startswith(str(new_root / 'valid'))
    np.testing.assert_array_equal(first['world'], new_expected['val'][0, :, :13])
    old = viewer.sample(opened['dataset_id'], 1)
    assert old['path'].startswith(str(old_root / 'processed' / 'data' / 'train'))
    np.testing.assert_array_equal(old['world'], old_expected['train'][0, :, :13])
    final = viewer.sample(opened['dataset_id'], 7)
    assert final['path'].startswith(str(new_root / 'test'))
    np.testing.assert_array_equal(final['world'], new_expected['test'][0, :, :13])


def test_plain_standard_files_use_entered_roots_and_sorted_file_order(tmp_path):
    roots = [tmp_path / 'first', tmp_path / 'second']
    for root in roots:
        (root / 'data').mkdir(parents=True)
    packed_fixture(1, 20).tofile(roots[0] / 'data/b.bin')
    packed_fixture(2, 0).tofile(roots[0] / 'data/a.bin')
    packed_fixture(1, 100).tofile(roots[1] / 'data/only.bin')
    config_path = config_fixture(tmp_path / 'config.yaml')
    viewer = DatasetViewer()
    opened = viewer.open('\n'.join(str(root) for root in roots), str(config_path), 'all')
    assert opened['count'] == 4 and opened['files'] == 3
    assert opened['roots'] == [str(root) for root in roots]
    sample = viewer.sample(opened['dataset_id'], 2)
    assert sample['file_index'] == 1 and sample['window_index'] == 0
    assert sample['dataset_idx'] == 0 and sample['path'].endswith('/first/data/b.bin')
    assert sample['world'][0][0] == 20
    sample = viewer.sample(opened['dataset_id'], 3)
    assert sample['dataset_idx'] == 1 and sample['world'][0][0] == 100
    with pytest.raises(ValueError, match='split'):
        viewer.open([str(roots[0])], str(config_path), 'train')


@pytest.mark.parametrize('index', [-1, 2, .5, '1', True, None])
def test_api_rejects_invalid_indices_and_unknown_dataset(tmp_path, index):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'assembled'
    assembled_fixture(root, cfg)
    client = create_app(tmp_path / 'gui').test_client()
    opened = open_fixture(client, root, config_fixture(tmp_path / 'config.yaml'))
    response = client.post('/api/dataset-viewer/sample', json={
        'dataset_id': opened['dataset_id'], 'index': index})
    assert response.status_code == 400 and response.get_json()['error']
    assert client.post('/api/dataset-viewer/sample', json={
        'dataset_id': 'unknown', 'index': 0}).status_code == 400


def test_api_rejects_config_mismatch_malformed_bins_and_nonfinite_sample(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'assembled'
    assembled_fixture(root, cfg)
    client = create_app(tmp_path / 'gui').test_client()
    mismatch = config_fixture(tmp_path / 'mismatch.yaml', dt=.02)
    assert client.post('/api/dataset-viewer/open', json={
        'roots': [str(root)], 'config_path': str(mismatch)}).status_code == 400
    config_path = config_fixture(tmp_path / 'config.yaml')
    plain = tmp_path / 'plain'
    plain.mkdir()
    (plain / 'broken.bin').write_bytes(b'123')
    assert client.post('/api/dataset-viewer/open', json={
        'roots': [str(plain)], 'config_path': str(config_path), 'split': 'all'}).status_code == 400
    rows = packed_fixture(1)
    rows[0, 3, 1] = np.nan
    rows.tofile(plain / 'broken.bin')
    opened = open_fixture(client, plain, config_path, 'all')
    response = client.post('/api/dataset-viewer/sample', json={
        'dataset_id': opened['dataset_id'], 'index': 0})
    assert response.status_code == 400 and 'non-finite' in response.get_json()['error']


def test_viewer_keeps_only_four_recently_used_datasets(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'assembled'
    assembled_fixture(root, cfg)
    config_path = config_fixture(tmp_path / 'config.yaml')
    viewer = DatasetViewer()
    first = viewer.open([str(root)], str(config_path))['dataset_id']
    second = viewer.open([str(root)], str(config_path))['dataset_id']
    viewer.open([str(root)], str(config_path))
    viewer.open([str(root)], str(config_path))
    viewer.sample(first, 0)
    viewer.open([str(root)], str(config_path))
    assert len(viewer.datasets) == 4
    assert viewer.sample(first, 0)['index'] == 0
    with pytest.raises(ValueError, match='expired'):
        viewer.sample(second, 0)


@pytest.mark.parametrize('suffix', ['.yaml', '.yml', '.json'])
def test_viewer_config_roundtrip_includes_ordered_directories_and_all_editable_fields(tmp_path, monkeypatch, suffix):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'configs' / ('viewer' + suffix)
    data = {'history_steps': 3, 'future_steps': 2, 'dt': .04}
    # Saving and loading config does not require the datasets to be mounted yet.
    expected_roots = [str(tmp_path / 'second'), str(tmp_path / 'first')]
    saved = save_viewer_config(path, ['second', 'first', 'second'], data, 'all')
    assert saved == {'config_path': str(path), 'roots': expected_roots,
                     'data': asdict(DataConfig(**data)), 'split': 'all'}
    assert load_viewer_config(path) == saved
    assert yaml.safe_load(path.read_text()) == {'datasets': expected_roots, 'data': data, 'split': 'all'}
    assert not (tmp_path / 'first').exists() and not (tmp_path / 'second').exists()
    if suffix == '.json':
        assert json.loads(path.read_text())['datasets'] == expected_roots


def test_load_viewer_training_config_resolves_relative_directories_and_top_level_takes_precedence(tmp_path):
    path = config_fixture(tmp_path / 'training.yaml')
    doc = yaml.safe_load(path.read_text())
    doc['training'] = {'datasets': ['second', 'first'], 'epochs': 50}
    path.write_text(yaml.safe_dump(doc))
    loaded = load_viewer_config(path)
    assert loaded['roots'] == [str(tmp_path / 'second'), str(tmp_path / 'first')]
    assert loaded['split'] == 'train'
    assert loaded['data']['history_steps'] == 3 and loaded['data']['stride'] == 2
    doc['datasets'] = ['top']
    doc['split'] = 'val'
    path.write_text(yaml.safe_dump(doc))
    assert load_viewer_config(path)['roots'] == [str(tmp_path / 'top')]
    assert load_viewer_config(path)['split'] == 'val'


def test_save_viewer_existing_training_config_preserves_unrelated_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'training.yaml'
    old_data = {'history_steps': 250, 'future_steps': 50, 'dt': .02,
                'stride': 75, 'shard_windows': 10000, 'features': 42, 'custom_data': {'keep': True}}
    doc = {'data': old_data, 'training': {'datasets': ['old'], 'epochs': 90, 'batch_size': 32},
           'model': {'hidden_size': 256}, 'commands': {'duration': 10}, 'notes': 'keep this'}
    path.write_text(yaml.safe_dump(doc))
    saved = save_viewer_config(path, ['real', 'sim'], {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'test')
    edited = yaml.safe_load(path.read_text())
    roots = [str(tmp_path / 'real'), str(tmp_path / 'sim')]
    assert edited['datasets'] == edited['training']['datasets'] == roots
    assert edited['training'] == {**doc['training'], 'datasets': roots}
    assert edited['data'] == {**old_data, 'history_steps': 3, 'future_steps': 2, 'dt': .04}
    for key in ('model', 'commands', 'notes'):
        assert edited[key] == doc[key]
    assert saved == load_viewer_config(path)
    assert saved['data']['stride'] == 75 and saved['data']['shard_windows'] == 10000


@pytest.mark.parametrize('name', ['manifest.json', 'dataset_build.json', 'schema.json', 'dataset_config.py'])
def test_save_viewer_config_cannot_overwrite_generated_metadata(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b'preserve generated metadata\n')
    before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    with pytest.raises(ValueError, match='new config filename'):
        save_viewer_config(path, ['dataset'], {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'train')
    assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == before


def test_save_viewer_config_protects_generated_metadata_through_symlink(tmp_path):
    metadata = tmp_path / 'schema.json'
    metadata.write_bytes(b'preserve metadata\n')
    alias = tmp_path / 'viewer.yaml'
    alias.symlink_to(metadata)
    with pytest.raises(ValueError, match='new config filename'):
        save_viewer_config(alias, ['dataset'], {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'train')
    assert alias.is_symlink() and metadata.read_bytes() == b'preserve metadata\n'


@pytest.mark.parametrize('change', [
    {'path': ''}, {'path': None}, {'path': 'missing/config'}, {'path': 'missing/config.txt'},
    {'roots': []}, {'roots': ['']}, {'roots': [None]}, {'split': 'valid'},
    {'data': {'dt': .04}}, {'data': {'history_steps': True, 'future_steps': 2, 'dt': .04}},
    {'data': {'history_steps': 3, 'future_steps': 0, 'dt': .04}},
    {'data': {'history_steps': 3, 'future_steps': 2, 'dt': float('nan')}},
])
def test_save_viewer_config_validates_before_creating_files(tmp_path, change):
    kwargs = {'path': str(tmp_path / 'missing' / 'viewer.yaml'), 'roots': ['dataset'],
              'data': {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'split': 'train', **change}
    if isinstance(kwargs['path'], str) and kwargs['path'].startswith('missing/'):
        kwargs['path'] = str(tmp_path / kwargs['path'])
    with pytest.raises(ValueError):
        save_viewer_config(**kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('text', ['data:\n  history_steps: [3\n', 'data: null\n', 'training: null\n'])
def test_save_viewer_config_rejects_malformed_existing_file_without_modification(tmp_path, text):
    path = tmp_path / 'viewer.yaml'
    path.write_text(text)
    before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    client = create_app(tmp_path / 'gui').test_client()
    response = client.post('/api/dataset-viewer/config/save', json={'path': str(path), 'roots': ['dataset'],
        'data': {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'split': 'train'})
    assert response.status_code == 400, response.get_json()
    assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == before


def test_load_viewer_config_infers_generated_roots_and_keeps_legacy_dimension_only_configs(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    flat = tmp_path / 'processed' / 'sim' / 'road'
    legacy = tmp_path / 'old' / 'road' / 'sim'
    assembled_fixture(flat, cfg, flat=True)
    assembled_fixture(legacy, cfg)
    for root, metadata in [(flat, flat / 'metadata'), (legacy, legacy / 'processed' / 'metadata')]:
        for name in ('dataset_config.py', 'schema.json'):
            loaded = load_viewer_config(metadata / name)
            assert loaded['roots'] == [str(root)]
            assert loaded['data']['history_steps'] == 3 and loaded['data']['future_steps'] == 2
    index = tmp_path / 'processed' / 'dataset_build.json'
    index.write_text(json.dumps({'config': asdict(cfg), 'output_layout': 'source_terrain_split_v1'}))
    assert load_viewer_config(index)['roots'] == [str(index.parent)]
    assert load_viewer_config(flat / 'manifest.json')['roots'] == [str(flat)]
    assert load_viewer_config(config_fixture(tmp_path / 'dimensions.yaml'))['roots'] == []
    plain = tmp_path / 'standard' / 'processed'
    (plain / 'metadata').mkdir(parents=True)
    (plain / 'data').mkdir()
    py = plain / 'metadata' / 'dataset_config.py'
    py.write_text('class ProcessedDatasetConfig:\n    history_horizon = 3\n    future_horizon = 3\n    control_dt = .04\n')
    assert load_viewer_config(py)['roots'] == [str(plain)]


def test_api_viewer_config_save_load_and_open_fills_roots_dimensions_and_split(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    roots = [tmp_path / 'processed' / 'real' / 'road', tmp_path / 'processed' / 'sim' / 'road']
    for root in roots:
        assembled_fixture(root, cfg, flat=True)
    client = create_app(tmp_path / 'gui').test_client()
    path = tmp_path / 'configs' / 'viewer.yaml'
    saved = client.post('/api/dataset-viewer/config/save', json={'path': str(path),
        'roots': [str(root) for root in roots],
        'data': {'history_steps': 3, 'future_steps': 2, 'dt': .04}, 'split': 'val'})
    assert saved.status_code == 200, saved.get_json()
    loaded = client.post('/api/dataset-viewer/config/load', json={'path': str(path)})
    assert loaded.status_code == 200 and loaded.get_json() == saved.get_json()
    opened = client.post('/api/dataset-viewer/open', json={'config_path': str(path)})
    assert opened.status_code == 200, opened.get_json()
    assert opened.json['roots'] == [str(root) for root in roots]
    assert opened.json['split'] == 'val' and opened.json['count'] == 2
    override = client.post('/api/dataset-viewer/open', json={'config_path': str(path), 'split': 'all'})
    assert override.status_code == 200 and override.json['count'] == 8
    assert override.json['split'] == 'all'


def test_api_inline_viewer_values_take_precedence_over_file_and_need_no_config(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'processed' / 'sim' / 'road'
    assembled_fixture(root, cfg, flat=True)
    client = create_app(tmp_path / 'gui').test_client()
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text('this: [is broken yaml\n')
    data = {'history_steps': 3, 'future_steps': 2, 'dt': .04}
    for config_path in (None, str(tmp_path / 'nonexistent.yaml'), str(invalid)):
        response = client.post('/api/dataset-viewer/open', json={'roots': [str(root)],
            'config_path': config_path, 'data': data, 'split': 'test'})
        assert response.status_code == 200, response.get_json()
        assert response.json['count'] == 1 and response.json['split'] == 'test'
        assert response.json['dt'] == .04
    for bad in ({'dt': .04}, {**data, 'history_steps': 2, 'future_steps': 3}, {**data, 'dt': .02}):
        response = client.post('/api/dataset-viewer/open', json={'roots': [str(root)], 'data': bad})
        assert response.status_code == 400
    unavailable = client.post('/api/dataset-viewer/open', json={'roots': [str(tmp_path / 'missing')], 'data': data})
    assert unavailable.status_code == 400 and 'does not exist' in unavailable.json['error']


def test_manual_relative_roots_resolve_the_same_for_save_and_inline_open(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    root = tmp_path / 'dataset'
    assembled_fixture(root, cfg, flat=True)
    data = {'history_steps': 3, 'future_steps': 2, 'dt': .04}
    path = tmp_path / 'configs' / 'viewer.yaml'
    saved = save_viewer_config(path, ['dataset'], data, 'train')
    opened = DatasetViewer().open(['dataset'], str(path), 'train', data)
    assert saved['roots'] == opened['roots'] == [str(root)]
    assert load_viewer_config(path)['roots'] == saved['roots']
