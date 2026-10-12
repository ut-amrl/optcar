"""Reference FKD slicing/selection over physical OptCar processed windows."""
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from optcar.dataloader_config import DataloaderConfig
from optcar.dataset import DataConfig, SCHEMA, pack_windows
from optcar.fkd_dataloader import FKDDataset
from optcar.train import split_dataset
from optcar.utils.math import euler_to_quat_torch


def packed(count, steps=7, history=4, offset=0):
    generator = torch.Generator().manual_seed(19)
    states = torch.randn(count, steps, 13, generator=generator)
    angles = torch.randn(count * steps, 3, generator=generator)
    states[..., 3:7] = euler_to_quat_torch(angles).reshape(count, steps, 4)
    states[..., 0] += offset
    states[:, 1::2, 3:7] *= -1
    actions = torch.randn(count, steps, 2, generator=generator)
    return pack_windows(states, actions, history, vehicle_id=12)


def processed(root, counts=(3, 4, 5), history=4, future=3, dt=.02, offset=0):
    config = DataConfig(history_steps=history, future_steps=future, dt=dt)
    expected, shards, episodes = {}, [], {}
    for split in ('train', 'val', 'test'):
        folder = root / ('valid' if split == 'val' else split)
        folder.mkdir(parents=True)
        episodes[split] = {'session': str(root) + '/' + split, 'split': split, 'shards': []}
        for file_idx, count in enumerate(counts):
            path = folder / f'{file_idx:010d}.bin'
            values = packed(count, history + future, history, offset + file_idx * 50)
            values.numpy().tofile(path)
            expected[str(path)] = values
            shards.append({'path': str(path.relative_to(root)), 'shape': list(values.shape), 'split': split})
    doc = {'schema': SCHEMA, 'config': asdict(config), 'source': 'real' if '/real/' in str(root) else 'sim',
           'terrain': root.name, 'episodes': episodes, 'processed_shards': shards,
           'provenance': {}, 'process_windows': True, 'flat_processed': True}
    (root / 'manifest.json').write_text(json.dumps(doc))
    return expected


def dataloader_config(paths, **changes):
    return DataloaderConfig(dataset_path=[str(path) for path in paths], num_runs=-1,
                            device='cpu', sim_dt=.02, **changes)





@pytest.mark.parametrize('preload', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_physical_rows_are_sliced_before_sample_construction(tmp_path, preload, dtype, oracle):
    root = tmp_path / 'processed' / 'sim' / 'road'
    expected = processed(root, counts=(2,))
    cfg = dataloader_config([root], dtype=dtype)
    actual = FKDDataset(cfg, future_horizon_steps=3, history_horizon_steps=4, preload_to_ram=preload)
    original_bytes = {path: Path(path).read_bytes() for path in expected}
    for index in range(2):
        prefix = f'fkd_sample.{index}.'
        want = {key[len(prefix):]: torch.from_numpy(value).to(dtype) if value.dtype.kind == 'f' else torch.from_numpy(value)
                for key, value in oracle.items() if key.startswith(prefix)}
        sample = actual[index]
        assert set(want) <= set(sample)
        for key, value in want.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(sample[key], value, atol=0, rtol=0)
            else:
                assert sample[key] == value
        assert sample['history'].shape == (4, 13)
        assert sample['future'].shape == sample['delta_future'].shape == (3, 13)
        assert sample['history_actions'].shape == (4, 2)
        assert sample['future_actions'].shape == (3, 2)
        assert sample['full_data'].shape == (6, 42)
        torch.testing.assert_close(sample['history'][-1], sample['future'][0], atol=0, rtol=0)
        torch.testing.assert_close(sample['delta_future'][0, :7], torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=dtype))
    assert actual.shapes == [(2, 7, 42)]
    assert all(Path(path).read_bytes() == value for path, value in original_bytes.items())


def test_250_history_50_future_uses_299_of_300_saved_rows(tmp_path):
    root = tmp_path / 'processed' / 'real' / 'grass'
    processed(root, counts=(2,), history=250, future=50)
    dataset = FKDDataset(dataloader_config([root]), 50, 250)
    sample = dataset[0]
    assert dataset.shapes == [(2, 300, 42)]
    assert sample['history'].shape == (250, 13)
    assert sample['future'].shape == (50, 13)
    assert sample['full_data'].shape == (299, 42)
    assert sample['history_actions'].shape == (250, 2)
    assert sample['future_actions'].shape == (50, 2)


@pytest.mark.parametrize('seed', [None, 0, 42])
def test_file_and_trajectory_selection_order(tmp_path, seed, oracle):
    roots = [tmp_path / 'processed' / 'sim' / terrain for terrain in ('grass', 'road')]
    for dataset_idx, root in enumerate(roots):
        processed(root, offset=dataset_idx * 100)
    cfg = dataloader_config(roots, num_trajectories=[4, 7], file_selection_seed=seed)
    cfg.num_runs = [2, 3]
    actual = FKDDataset(cfg, 3, 4)
    key = f'selection.seed_{seed}'
    file_names = {None: [0,1,0,1,2], 0: [2,0,1,2,0], 42: [0,2,1,0,2]}[seed]
    files = [str(roots[folder] / 'train' / f'{name:010d}.bin')
             for folder, name in zip(oracle[key + '.file_map'], file_names)]
    indices = oracle[key + '.indices'].tolist()
    cumulative = [0, *np.cumsum([Path(file).stat().st_size // (7*42*4) for file in files]).tolist()][:-1]
    assert actual.files == files
    assert actual.dataset_file_mapping == oracle[key + '.file_map'].tolist()
    assert actual._global_indices == indices
    assert len(actual) == 11
    for index, global_idx in enumerate(indices):
        file_idx = bisect_right(cumulative, global_idx) - 1
        row_idx = global_idx - cumulative[file_idx]
        rows = np.fromfile(files[file_idx], np.float32).reshape(-1, 7, 42)
        torch.testing.assert_close(actual[index]['full_data'], torch.from_numpy(rows[row_idx, :6]), atol=0, rtol=0)


def test_recursive_parent_roots_keep_main_train_and_test_disjoint_and_validation_matches_reference(tmp_path):
    parent = tmp_path / 'processed'
    for source, terrain in [('real', 'road'), ('real', 'grass'), ('sim', 'road')]:
        processed(parent / source / terrain, counts=(2,))
    cfg = dataloader_config([parent / 'real', parent / 'sim', parent],
        test_dataset_path=[str(parent)], test_num_runs=-1)
    main, test = FKDDataset(cfg, 3, 4), FKDDataset(cfg, 3, 4, is_test_set=True)
    assert len(main) == len(test) == 6
    assert set(main.files).isdisjoint(test.files)
    assert all('/train/' in path for path in main.files)
    assert all('/test/' in path for path in test.files)
    train, validation, unused, residue = split_dataset(main, train_ratio=.7, val_ratio=.3, test_ratio=0, seed=42)
    assert len(train) == 4 and len(validation) == 1 and len(unused) == 1 and len(residue) == 0
    assert train.dataset is validation.dataset is main
    assert set(train.indices).isdisjoint(validation.indices)


def test_plain_metadata_and_manifest_metadata_can_have_different_physical_lengths(tmp_path):
    optcar = tmp_path / 'processed' / 'sim' / 'road'
    processed(optcar, counts=(2,))
    original = tmp_path / 'plain'
    (original / 'data').mkdir(parents=True)
    (original / 'metadata').mkdir()
    packed(2, steps=6).numpy().tofile(original / 'data' / 'a.bin')
    (original / 'metadata' / 'dataset_config.py').write_text(
        'class ProcessedDatasetConfig:\n    history_horizon = 4\n    future_horizon = 3\n'
        '    control_dt = .02\n    feature_dim = 42\n    data_dtype = "float32"\n')
    dataset = FKDDataset(dataloader_config([original, optcar]), 3, 4)
    assert dataset.shapes == [(2, 6, 42), (2, 7, 42)]
    assert len(dataset) == 4
    for index in range(4):
        assert dataset[index]['full_data'].shape == (6, 42)
    cfg = dataloader_config([original], test_dataset_path=str(original), test_num_runs=-1)
    with pytest.raises(ValueError, match='overlap'):
        FKDDataset(cfg, 3, 4, is_test_set=True)


def test_variable_horizon_mask_matches_reference_rng_and_keeps_full_future(tmp_path):
    root = tmp_path / 'processed' / 'sim' / 'road'
    expected = processed(root, counts=(1,))
    cfg = dataloader_config([root], variable_horizon=True)
    dataset = FKDDataset(cfg, 3, 4)
    torch.manual_seed(23)
    horizon = torch.randint(1, 4, (1,)).item()
    torch.manual_seed(23)
    actual = dataset[0]
    torch.testing.assert_close(actual['future_horizon_mask'], torch.arange(3) < horizon)
    assert actual['future'].shape == (3, 13)


def test_preload_uses_16_threads_and_worker_memmaps_are_reopened(tmp_path, monkeypatch):
    root = tmp_path / 'processed' / 'real' / 'grass'
    processed(root, counts=(1,))
    workers = []
    def pool(*args, **kwargs):
        workers.append(kwargs['max_workers'])
        return ThreadPoolExecutor(*args, **kwargs)
    monkeypatch.setattr('optcar.fkd_dataloader.ThreadPoolExecutor', pool)
    cfg = dataloader_config([root])
    ram = FKDDataset(cfg, 3, 4, preload_to_ram=True)
    assert workers == [16]
    mmap = FKDDataset(cfg, 3, 4)
    assert all(item is not None for item in mmap._mmaps)
    state = mmap.__getstate__()
    assert state['_mmaps'] == [None]
    restored = FKDDataset.__new__(FKDDataset)
    restored.__setstate__(state)
    torch.testing.assert_close(restored[0]['full_data'], ram[0]['full_data'])
    batches = list(DataLoader(mmap, batch_size=1, num_workers=1))
    assert batches[0]['future'].shape == (1, 3, 13)


@pytest.mark.parametrize('changes', [{'history_horizon_steps': 3}, {'future_horizon_steps': 5}, {'sim_dt': .04}])
def test_incompatible_model_anchor_horizon_or_timestep_is_rejected(tmp_path, changes):
    root = tmp_path / 'processed' / 'sim' / 'road'
    expected = processed(root, counts=(1,))
    cfg = dataloader_config([root])
    cfg.sim_dt = changes.get('sim_dt', cfg.sim_dt)
    before = {path: Path(path).read_bytes() for path in expected}
    with pytest.raises(ValueError, match='history|shorter|timestep'):
        FKDDataset(cfg, changes.get('future_horizon_steps', 3), changes.get('history_horizon_steps', 4))
    assert all(Path(path).read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('flag', ['include_dbm_params', 'is_cmd_vel_in_data'])
def test_extended_layout_flags_are_rejected_for_standard_optcar_bins(tmp_path, flag):
    cfg = dataloader_config([tmp_path], **{flag: True})
    with pytest.raises(ValueError, match='42-column'):
        FKDDataset(cfg, 3, 4)
