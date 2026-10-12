"""Processed-window indexing and sample parity with WheeledLab's loader."""
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest
import torch

from optcar.dataset import DataConfig, FKDDataset, SCHEMA, model_sample, pack_windows
from optcar.utils.math import euler_to_quat_torch
from optcar.dataloader_config import DataloaderConfig


def windows(count=5):
    generator = torch.Generator().manual_seed(13)
    states = torch.randn(count, 7, 13, generator=generator)
    states[..., 3:7] = euler_to_quat_torch(torch.randn(count * 7, 3, generator=generator)).reshape(count, 7, 4)
    states[:, 1::2, 3:7] *= -1
    actions = torch.randn(count, 7, 2, generator=generator)
    return pack_windows(states, actions, 4, vehicle_id=9)


def manifest_dataset(root, values, split_counts=(2, 2, 1), session="recording", flat=False):
    root.mkdir(parents=True)
    shards, episodes, offset = [], {}, 0
    for split, count in zip(("train", "val", "test"), split_counts):
        if not count:
            continue
        folder = root / ("valid" if split == "val" else split) if flat else root / "processed" / "data" / split
        file = folder / "processed_0000000000.bin"
        file.parent.mkdir(parents=True)
        array = values[offset:offset + count].numpy()
        array.tofile(file)
        shards.append({"path": str(file.relative_to(root)), "shape": list(array.shape), "split": split})
        episodes[split] = {"session": f"{session}-{split}", "split": split, "shards": []}
        offset += count
    config = DataConfig(history_steps=4, future_steps=3)
    document = {"schema": SCHEMA, "process_windows": True, "source": "sim", "terrain": "road",
                "config": asdict(config), "provenance": {}, "episodes": episodes, "processed_shards": shards}
    (root / "manifest.json").write_text(json.dumps(document))
    return config, shards


def test_manifest_subfolders_all_splits_and_overlapping_roots(tmp_path):
    values = windows()
    root = tmp_path / "dataset"
    config, shards = manifest_dataset(root, values)
    assert len(FKDDataset(root, config)) == 2
    assert len(FKDDataset(root / "processed", config, "all")) == 5
    train = root / "processed" / "data" / "train"
    dataset = FKDDataset([train, root, root / "processed"], config, "all")
    assert len(dataset) == 5
    for index in range(5):
        torch.testing.assert_close(dataset.load_window(index), values[index], atol=0, rtol=0)
    assert dataset.locate_window(1)["window_index"] == 1
    assert dataset.locate_window(2)["file_index"] == 1
    assert dataset.locate_window(2)["window_index"] == 0
    assert dataset.locate_window(2)["dataset_idx"] == 1
    assert dataset.locate_window(4)["split"] == "test"
    assert dataset.locate_window(4)["path"] == str(root / shards[2]["path"])
    with pytest.raises(ValueError, match="No val windows"):
        FKDDataset(train, config, "val")


def test_flat_source_terrain_layout_keeps_recorded_validation_split(tmp_path):
    values = windows()
    root = tmp_path / "processed" / "sim" / "road"
    config, shards = manifest_dataset(root, values, flat=True)
    assert [shard["path"].split("/")[0] for shard in shards] == ["train", "valid", "test"]
    assert len(FKDDataset(root.parent.parent, config, "all")) == 5
    validation = FKDDataset(root / "valid", config, "val")
    assert len(validation) == 2
    assert validation.locate_window(0)["split"] == "val"
    assert validation.locate_window(0)["path"] == str(root / shards[1]["path"])
    for index, value in enumerate(values[2:4]):
        torch.testing.assert_close(validation.load_window(index), value, atol=0, rtol=0)
    assert len(FKDDataset(root / "valid", config, "all", allow_wheeledlab=True)) == 2
    with pytest.raises(ValueError, match="No train windows"):
        FKDDataset(root / "valid", config, "train", allow_wheeledlab=True)


def test_mixed_legacy_and_flat_manifest_roots_preserve_order_and_deduplicate(tmp_path):
    old_values, new_values = windows(), windows()
    new_values[:, :, 0] += 100
    old_root = tmp_path / "old" / "road" / "sim"
    new_root = tmp_path / "new" / "processed" / "sim" / "road"
    config, _ = manifest_dataset(old_root, old_values, session="old")
    manifest_dataset(new_root, new_values, session="new", flat=True)
    dataset = FKDDataset([new_root / "valid", old_root / "processed", new_root, old_root], config, "all")
    expected = torch.cat((new_values[2:4], old_values, new_values[:2], new_values[4:]))
    assert len(dataset) == 10 and len(dataset.shards) == 6
    for index, value in enumerate(expected):
        torch.testing.assert_close(dataset.load_window(index), value, atol=0, rtol=0)
    assert dataset.locate_window(0)["split"] == "val"
    assert dataset.locate_window(2)["path"].startswith(str(old_root))
    assert dataset.locate_window(7)["path"].startswith(str(new_root / "train"))


def test_wheeledlab_directory_order_sorted_files_and_opt_in(tmp_path):
    values = windows()
    second, first = tmp_path / "second", tmp_path / "first" / "data"
    second.mkdir(); first.mkdir(parents=True)
    values[2:4].numpy().tofile(first / "b.bin")
    values[1:2].numpy().tofile(first / "a.bin")
    values[0:1].numpy().tofile(second / "x.bin")
    values[4:5].numpy().tofile(second / "z.bin")
    config = DataConfig(history_steps=4, future_steps=3)
    dataset = FKDDataset([second, first.parent, first], config, "all", allow_wheeledlab=True)
    expected = values[[0, 4, 1, 2, 3]]
    assert len(dataset) == 5
    assert [file[0].name for file in dataset.shards] == ["x.bin", "z.bin", "a.bin", "b.bin"]
    assert dataset.dataset_file_mapping == [0, 0, 1, 1]
    for index, value in enumerate(expected):
        torch.testing.assert_close(dataset.load_window(index), value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="no recorded split"):
        FKDDataset(second, config, allow_wheeledlab=True)
    with pytest.raises(ValueError, match="No all windows"):
        FKDDataset(second, config, "all")


def test_training_contract_bounds_and_viewing_cannot_change_bins(tmp_path):
    values = windows()
    config, _ = manifest_dataset(tmp_path / "dataset", values)
    dataset = FKDDataset(tmp_path / "dataset", config, "all")
    originals = {file: file.read_bytes() for file, *_ in dataset.shards}
    actual = dataset[np.int64(2)]
    expected = {**model_sample(values[2], 4), "source": "sim", "terrain": "road"}
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
        else:
            assert actual[key] == value
    for method in (dataset.load_window, dataset.locate_window, dataset.get_fkd_sample, dataset.__getitem__):
        for index in (-1, len(dataset)):
            with pytest.raises(IndexError):
                method(index)
        with pytest.raises(TypeError):
            method(1.0)
    torch.testing.assert_close(dataset.load_window(torch.tensor(2)), values[2], atol=0, rtol=0)
    dataset.load_window(2).zero_()
    dataset.get_fkd_sample(2)["full_data"].zero_()
    torch.testing.assert_close(dataset.load_window(2), values[2], atol=0, rtol=0)
    assert all(file.read_bytes() == original for file, original in originals.items())
    assert not dataset.__getstate__()["cache"]


def test_malformed_plain_bin_does_not_get_truncated(tmp_path):
    np.zeros(43, dtype=np.float32).tofile(tmp_path / "invalid.bin")
    with pytest.raises(ValueError, match="Invalid binary size"):
        FKDDataset(tmp_path, DataConfig(history_steps=4, future_steps=3), "all", allow_wheeledlab=True)


def test_fkd_sample_matches_original_wheeledlab_getitem(tmp_path):
    source_root = Path(os.environ.get("WHEELEDLAB_SOURCE", Path(__file__).resolve().parents[2] / "WheeledLab/source"))
    source = source_root / "wheeledlab_data/wheeledlab_data/dataloaders/FKD/dataloader.py"
    if not source.is_file():
        pytest.skip("Local WheeledLab source is unavailable")
    tree = ast.parse(source.read_text())
    loader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FKDDataset")
    method = next(node for node in loader.body if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
    namespace = {"torch": torch, "np": np, "Dict": Dict}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    values = windows()
    config, _ = manifest_dataset(tmp_path / "dataset", values)
    dataset = FKDDataset(tmp_path / "dataset", config, "all")
    reference = SimpleNamespace(_global_indices=None, preload_to_ram=True, _data_arrays=[values.numpy()],
        _find_file_and_agent_idx=lambda index: (0, index), dtype=torch.float32, history_horizon_steps=4,
        dataset_file_mapping=[0], dl_config=SimpleNamespace(is_cmd_vel_in_data=False), _dbm_dim=0,
        dummy_hmap_torch=torch.zeros((1, 1)))
    for index in range(5):
        expected = namespace["__getitem__"](reference, index)
        actual = dataset.get_fkd_sample(index)
        assert set(expected) <= set(actual)
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
            else:
                assert actual[key] == value
        assert actual["history"].shape == (4, 13)
        assert actual["future"].shape == (4, 13)  # current + three transitions
        torch.testing.assert_close(actual["history"][-1], actual["future"][0], atol=0, rtol=0)
        torch.testing.assert_close(actual["full_delta_states"], values[index, :, 26:39], atol=0, rtol=0)


def test_recorded_session_leak_checks_remain_active(tmp_path):
    values = windows()
    config, _ = manifest_dataset(tmp_path / "dataset", values)
    path = tmp_path / "dataset" / "manifest.json"
    document = json.loads(path.read_text())
    document["episodes"]["val"]["session"] = document["episodes"]["train"]["session"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Session leakage"):
        FKDDataset(path.parent / "processed", config, "all", allow_wheeledlab=True)


def test_manifest_quaternion_order_must_match_wheeledlab_wxyz(tmp_path):
    config, _ = manifest_dataset(tmp_path / "dataset", windows())
    path = tmp_path / "dataset" / "manifest.json"
    document = json.loads(path.read_text())
    document["schema"]["quaternion_order"] = "xyzw"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="quaternion order.*expected wxyz"):
        FKDDataset(path.parent / "processed", config, "all", allow_wheeledlab=True)
    del document["schema"]["quaternion_order"]
    path.write_text(json.dumps(document))
    assert len(FKDDataset(path.parent, config, "all")) == 5


@pytest.mark.parametrize('seed', [None, 0, 42])
def test_file_and_trajectory_limits_match_wheeledlab_methods(tmp_path, seed):
    """Run WheeledLab's actual selection methods against the same saved files."""
    source_root = Path(os.environ.get('WHEELEDLAB_SOURCE', Path(__file__).resolve().parents[2] / 'WheeledLab/source'))
    source = source_root / 'wheeledlab_data/wheeledlab_data/dataloaders/FKD/dataloader.py'
    if not source.is_file():
        pytest.skip('Local WheeledLab source is unavailable')
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FKDDataset')
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_load_file_names', '_apply_trajectory_limits')]
    from bisect import bisect_right
    from typing import List
    namespace = {'torch': torch, 'Path': Path, 'Dict': Dict, 'List': List, 'bisect_right': bisect_right}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    roots = [tmp_path / 'grass', tmp_path / 'road']
    values = windows(12)
    for dataset_idx, root in enumerate(roots):
        (root / 'data').mkdir(parents=True)
        for file_idx, (start, count) in enumerate([(0, 3), (3, 4), (7, 5)]):
            rows = values[start:start + count].clone()
            rows[:, :, 0] += dataset_idx * 100 + file_idx * 10
            rows.numpy().tofile(root / 'data' / f'{file_idx}.bin')
    cfg = DataloaderConfig(dataset_path=[str(root) for root in roots], num_runs=[2, 3],
                           num_trajectories=[4, 7], file_selection_seed=seed)
    reference = SimpleNamespace(dataset_paths=cfg.dataset_path, num_runs_list=cfg.num_runs,
                                num_trajectories_list=cfg.num_trajectories, dl_config=cfg,
                                files=[], dataset_file_mapping=[], _global_indices=None)
    namespace['_load_file_names'](reference)
    reference.file_traj_counts = [Path(file).stat().st_size // (7 * 42 * 4) for file in reference.files]
    reference.cumulative_traj_counts = [0, *np.cumsum(reference.file_traj_counts).tolist()][:-1]
    namespace['_apply_trajectory_limits'](reference)
    config = DataConfig(history_steps=4, future_steps=3)
    actual = FKDDataset(roots, config, 'all', allow_wheeledlab=True, dl_config=cfg)
    assert [str(shard[0]) for shard in actual.shards] == reference.files
    assert actual._global_indices == reference._global_indices
    assert len(actual) == 11
    for index, global_index in enumerate(reference._global_indices):
        file_idx = bisect_right(reference.cumulative_traj_counts, global_index) - 1
        window_idx = global_index - reference.cumulative_traj_counts[file_idx]
        rows = np.fromfile(reference.files[file_idx], dtype=np.float32).reshape(-1, 7, 42)
        torch.testing.assert_close(actual.load_window(index), torch.from_numpy(rows[window_idx]), atol=0, rtol=0)


def test_trajectory_caps_all_zero_and_test_config(tmp_path):
    roots = [tmp_path / 'grass', tmp_path / 'road']
    for root in roots:
        root.mkdir()
        windows(5).numpy().tofile(root / 'one.bin')
    config = DataConfig(history_steps=4, future_steps=3)
    cfg = DataloaderConfig(dataset_path=[str(root) for root in roots], num_runs=-1,
                           num_trajectories=[0, 3], test_dataset_path=[str(roots[1])],
                           test_num_runs=-1, test_num_trajectories=[2])
    data = FKDDataset(roots, config, 'all', allow_wheeledlab=True, dl_config=cfg)
    assert len(data) == 3
    assert [row['selected'] for row in data.selection_summary] == [0, 3]
    assert all(data.locate_window(i)['dataset_idx'] == 1 for i in range(3))
    test = FKDDataset([roots[1]], config, 'all', allow_wheeledlab=True, dl_config=cfg, is_test_set=True)
    assert len(test) == 2
    cfg.num_trajectories = [0, 0]
    with pytest.raises(ValueError, match='No trajectories selected'):
        FKDDataset(roots, config, 'all', allow_wheeledlab=True, dl_config=cfg)
    cfg.num_trajectories = [-1, None]
    data = FKDDataset(roots, config, 'all', allow_wheeledlab=True, dl_config=cfg)
    assert len(data) == 10 and data._global_indices is None
    cfg.num_runs = [0, -1]
    data = FKDDataset(roots, config, 'all', allow_wheeledlab=True, dl_config=cfg)
    assert len(data) == 5 and data.selection_summary[0]['files'] == 0


def test_training_terrain_sampler_respects_selected_window_indices(tmp_path):
    from optcar.dataset import TerrainSampler
    first, second = tmp_path / 'grass', tmp_path / 'road'
    config, _ = manifest_dataset(first, windows(), session='grass')
    manifest_dataset(second, windows(), session='road')
    doc = json.loads((first / 'manifest.json').read_text())
    doc['terrain'] = 'grass'
    (first / 'manifest.json').write_text(json.dumps(doc))
    cfg = DataloaderConfig(dataset_path=[str(first), str(second)], num_runs=-1,
                           num_trajectories=[0, 2], file_selection_seed=42)
    data = FKDDataset([first, second], config, 'all', dl_config=cfg)
    sampler = TerrainSampler(data, samples=50)
    for index in sampler:
        assert 0 <= index < len(data)
        assert data.locate_window(index)['terrain'] == 'road'
