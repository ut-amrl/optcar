"""Python config factories, selections and processed browsing."""
import ast
from dataclasses import fields
import importlib.util
from pathlib import Path

import pytest
import torch

from optcar.dataloader_config import DataloaderConfig
from optcar.gui.dataloader_setup import (config_values, load_python_setup, read_python_loader,
                                         save_python_setup, validate_loader_config)
from optcar.gui.dataset_viewer import load_viewer_config
from optcar.gui.server import create_app
from test_dataset_viewer import packed_fixture, assembled_fixture
from optcar.dataset import DataConfig


def fixture_setup(tmp_path):
    roots = [tmp_path / 'grass', tmp_path / 'road']
    for offset, root in enumerate(roots):
        root.mkdir()
        packed_fixture(5, offset * 100).tofile(root / 'a.bin')
        packed_fixture(3, offset * 100 + 20).tofile(root / 'b.bin')
    values = config_values(DataloaderConfig(dataset_path=[str(root) for root in roots],
                    num_runs=[-1, -1], num_trajectories=[3, 4],
                    test_dataset_path=[str(roots[1])], test_num_runs=[1],
                    test_num_trajectories=[2], sim_dt=.04, file_selection_seed=42, device='cpu'))
    return roots, values, {'history_steps': 3, 'future_steps': 2, 'dt': .04}


def test_python_export_is_importable_and_roundtrips_standard_fields(tmp_path):
    roots, values, data = fixture_setup(tmp_path)
    path = tmp_path / 'configs' / 'dconfig.py'
    saved = save_python_setup(str(path), values, data)
    assert load_viewer_config(path) == saved
    spec = importlib.util.spec_from_file_location('exported_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert config_values(module.dataloader_config()) == values
    assert module.viewer_data['history_steps'] == 3
    assert 'from optcar.dataloader_config import DataloaderConfig' in path.read_text()
    tree = ast.parse(path.read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert {keyword.arg for keyword in factory.body[0].value.keywords} == {field.name for field in fields(DataloaderConfig)}
    values['num_trajectories'] = [1, 2]
    save_python_setup(str(path), values, data)
    assert load_python_setup(path)['dl_config']['num_trajectories'] == [1, 2]


def test_read_actual_style_factory_never_executes_imports_or_code(tmp_path):
    marker = tmp_path / 'executed'
    path = tmp_path / 'dconfig.py'
    path.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
                    'from external_package import DataloaderConfig\n'
                    'import torch\n'
                    'def dataloader_config():\n'
                    '    return DataloaderConfig(dataset_path=["../grass"], num_runs=-1,\n'
                    '        num_trajectories=[10000], test_dataset_path=[], test_num_runs=[],\n'
                    '        test_num_trajectories=[], sim_dt=0.02, dtype=torch.float32, device="cpu")\n')
    loaded = load_python_setup(path)
    assert loaded['dl_config']['num_trajectories'] == [10000]
    assert loaded['roots'] == [str(tmp_path.parent / 'grass')]
    assert loaded['data'] is None
    assert not marker.exists()
    path.write_text('def dataloader_config():\n    return DataloaderConfig(dataset_path=calculate_paths())\n')
    with pytest.raises(ValueError, match='literal'):
        read_python_loader(path)


def test_config_infers_horizons_from_processed_group_metadata(tmp_path):
    group = tmp_path / 'grass'
    assembled_fixture(group, DataConfig(history_steps=3, future_steps=2, dt=.04))
    path = tmp_path / 'dconfig.py'
    values = config_values(DataloaderConfig(dataset_path=[str(group / 'processed' / 'data' / 'train')],
                                            num_runs=-1, sim_dt=.04))
    save_python_setup(str(path), values)
    loaded = load_python_setup(path)
    assert loaded['data']['history_steps'] == 3 and loaded['data']['future_steps'] == 2
    client = create_app(tmp_path / 'gui').test_client()
    opened = client.post('/api/dataset-viewer/open', json={'config_path': str(path)})
    assert opened.status_code == 200, opened.get_json()
    assert opened.get_json()['count'] == 2


def test_api_mixed_caps_test_set_and_shuffle_preserve_selected_windows(tmp_path):
    roots, values, data = fixture_setup(tmp_path)
    client = create_app(tmp_path / 'gui').test_client()
    response = client.post('/api/dataset-viewer/open', json={'dl_config': values, 'data': data, 'split': 'all'})
    assert response.status_code == 200, response.get_json()
    opened = response.get_json()
    assert opened['count'] == 7
    assert [entry['selected'] for entry in opened['selection']] == [3, 4]
    identifier = opened['dataset_id']
    def sample(index):
        result = client.post('/api/dataset-viewer/sample', json={'dataset_id': identifier, 'index': index})
        assert result.status_code == 200, result.get_json()
        return result.get_json()
    original = [sample(i) for i in range(7)]
    shuffle = client.post('/api/dataset-viewer/shuffle', json={'dataset_id': identifier, 'seed': 7})
    assert shuffle.status_code == 200
    perm = torch.randperm(7, generator=torch.Generator().manual_seed(7)).tolist()
    for index, source in enumerate(perm):
        result = sample(index)
        assert result['world'] == original[source]['world']
        assert result['path'] == original[source]['path']
        assert result['dataset_index'] == source and result['shuffled']
    test = client.post('/api/dataset-viewer/open', json={'dl_config': values, 'data': data, 'split': 'all', 'is_test_set': True})
    assert test.status_code == 200 and test.get_json()['count'] == 2
    assert test.get_json()['roots'] == [str(roots[1])]
    saved = client.post('/api/dataset-viewer/config/save', json={'path': str(tmp_path / 'dconfig.py'), 'dl_config': values, 'data': data})
    assert saved.status_code == 200, saved.get_json()
    reopened = client.post('/api/dataset-viewer/open', json={'config_path': saved.get_json()['config_path']})
    assert reopened.status_code == 200 and reopened.get_json()['count'] == 7


@pytest.mark.parametrize('change', [
    {'num_trajectories': [1]}, {'num_runs': [True, -1]}, {'num_trajectories': [1.5, 2]},
    {'file_selection_seed': -1}, {'dtype': 'float64'}, {'sim_dt': 0}, {'debug': 'false'},
    {'unexpected': True}, {'num_runs': -2}, {'plot_val_epoch_freq': 0},
])
def test_loader_config_validation_happens_before_writing(tmp_path, change):
    _, values, data = fixture_setup(tmp_path)
    path = tmp_path / 'dconfig.py'
    with pytest.raises(ValueError):
        save_python_setup(str(path), {**values, **change}, data)
    assert not path.exists()


def test_export_protects_metadata_and_unrelated_python(tmp_path):
    _, values, data = fixture_setup(tmp_path)
    metadata = tmp_path / 'dataset_config.py'
    metadata.write_text('class ProcessedDatasetConfig: pass\n')
    alias = tmp_path / 'alias.py'
    alias.symlink_to(metadata)
    unrelated = tmp_path / 'source.py'
    unrelated.write_text('print("do not execute or replace")\n')
    for path in (metadata, alias, unrelated):
        before = path.read_bytes()
        with pytest.raises(ValueError):
            save_python_setup(str(path), values, data)
        assert path.read_bytes() == before


def test_processed_explorer_lists_directories_and_bin_window_counts(tmp_path):
    roots, _, data = fixture_setup(tmp_path)
    client = create_app(tmp_path / 'gui').test_client()
    root = client.post('/api/dataset-viewer/browse', json={'path': str(tmp_path), 'data': data}).get_json()
    assert {'grass', 'road'} <= {folder['name'] for folder in root['folders']}
    assert root['bins'] == []
    (roots[0] / 'bad.bin').write_bytes(b'wrong size')
    folder = client.post('/api/dataset-viewer/browse', json={'path': str(roots[0]), 'data': data}).get_json()
    assert folder['trajectories'] == 8
    assert [(bin['name'], bin['trajectories']) for bin in folder['bins'] if not bin['error']] == [('a.bin', 5), ('b.bin', 3)]
    assert next(bin for bin in folder['bins'] if bin['name'] == 'bad.bin')['error']
