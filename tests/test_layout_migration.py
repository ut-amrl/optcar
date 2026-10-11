"""Rename existing processed datasets without rebuilding any trajectory."""
from dataclasses import asdict
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from optcar.dataset import DataConfig, DatasetWriter, FKDDataset
from optcar.gui.dataset_viewer import read_dataset_config


@pytest.fixture
def migration():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reorganize_processed_layout.py"
    spec = importlib.util.spec_from_file_location("optcar_layout_migration_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree_contents(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*"))}


def file_identity(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def old_dataset(tmp_path):
    root = tmp_path / "output" / "processed"
    cfg = DataConfig(history_steps=3, future_steps=2, dt=.04, stride=2, shard_windows=2)
    manifests, inputs, bins = {}, [], {}
    for terrain, source, counts, offset in [("road", "sim", (3, 2, 1), 10),
                                             ("grass", "real", (2, 1, 1), 30)]:
        group = root / terrain / source
        provenance = {"inputs": [], "calibrations": [], "label": "preserve original provenance"}
        raw_files = {}
        for split in ("train", "val", "test"):
            raw = tmp_path / "raw" / terrain / source / (split + ".bin")
            raw.parent.mkdir(parents=True, exist_ok=True)
            np.zeros((1, 8, 15), dtype=np.float32).tofile(raw)
            raw_files[split] = raw
            entry = {"path": str(raw), "terrain": terrain, "source": source, "split": split}
            provenance["inputs"].append(entry)
            inputs.append(entry)
        with DatasetWriter(group, cfg, source, terrain, provenance) as writer:
            for split, count in zip(("train", "val", "test"), counts):
                values = torch.zeros((count, cfg.window_steps, 42))
                values[..., 0] = torch.arange(count)[:, None] + offset
                values[..., 3] = values[..., 16] = values[..., 29] = 1
                values[..., 39] = .75
                values[..., 40] = -.2
                values[..., 41] = offset
                writer.write_window_batches(f"{terrain}_{source}_{split}", [values], split,
                    f"{terrain}_{source}_{split}_session", raw_files[split], [1, 8, 15])
        path = group / "manifest.json"
        manifests[group] = json.loads(path.read_text())
        for shard in manifests[group]["processed_shards"]:
            old = group / shard["path"]
            filename = "valid" if shard["split"] == "val" else shard["split"]
            new = root / source / terrain / filename / old.name
            bins[old] = {"new": new, "bytes": old.read_bytes(), "identity": file_identity(old),
                         "shape": shard["shape"], "split": shard["split"]}
    index = {"config": asdict(cfg), "bins": inputs, "seed": 42, "fractions": [.8, .1, .1]}
    (root / "dataset_build.json").write_text(json.dumps(index))
    (root / ".builder.lock").touch()
    return root, cfg, manifests, bins, index


def test_dry_run_is_read_only_and_reports_existing_bins(migration, tmp_path):
    root, _, _, bins, _ = old_dataset(tmp_path)
    original = tree_contents(tmp_path)
    identities = {path: file_identity(path) for path in bins}
    result = migration.reorganize(root, apply=False)
    assert result["status"] == "planned"
    assert (result["bins"], result["windows"], result["bytes"]) == (
        len(bins), 10, sum(len(item["bytes"]) for item in bins.values()))
    assert tree_contents(tmp_path) == original
    assert {path: file_identity(path) for path in bins} == identities


def test_apply_preserves_bins_and_loader_contract_without_regeneration(migration, tmp_path):
    root, cfg, documents, bins, index = old_dataset(tmp_path)
    raw_original = tree_contents(tmp_path / "raw")
    writer_locks = {group: file_identity(group / ".writer.lock") for group in documents}
    builder_lock = file_identity(root / ".builder.lock")
    result = migration.reorganize(root, apply=True)
    assert result["status"] == "applied"
    assert result["bins"] == len(bins) == 7 and result["windows"] == 10
    assert result["bytes"] == sum(len(item["bytes"]) for item in bins.values())
    assert set(root.rglob("*.bin")) == {item["new"] for item in bins.values()}
    for old, item in bins.items():
        assert not old.exists()
        assert item["new"].read_bytes() == item["bytes"]
        assert file_identity(item["new"]) == item["identity"]
    assert tree_contents(tmp_path / "raw") == raw_original
    for old_group, original in documents.items():
        new_group = root / original["source"] / original["terrain"]
        assert file_identity(new_group / ".writer.lock") == writer_locks[old_group]
        updated = json.loads((new_group / "manifest.json").read_text())
        assert updated["flat_processed"] is True
        assert updated["config"] == original["config"]
        assert updated["provenance"] == original["provenance"]
        assert updated["episodes"] == original["episodes"]
        assert updated["schema"] == original["schema"]
        expected_shards = [{**shard, "path": str(bins[old_group / shard["path"]]["new"].relative_to(new_group))}
                           for shard in original["processed_shards"]]
        assert updated["processed_shards"] == expected_shards
        schema = json.loads((new_group / "metadata" / "schema.json").read_text())
        assert schema["features"] == 42 and schema["quaternion_order"] == "wxyz"
        loaded = read_dataset_config(new_group / "metadata" / "dataset_config.py")
        assert (loaded.history_steps, loaded.future_steps, loaded.dt) == (3, 2, .04)
        assert not (new_group / "processed").exists()
        assert not (root / original["terrain"]).exists()
    expected_index = {**index, "output_layout": "source_terrain_split_v1"}
    assert json.loads((root / "dataset_build.json").read_text()) == expected_index
    assert file_identity(root / ".builder.lock") == builder_lock
    backup = Path(result["backup_root"])
    assert backup.is_dir() and not backup.is_relative_to(root)
    assert not list(backup.rglob("manifest.json"))
    assert list(backup.rglob("manifest.json.original"))
    assert any(json.loads(path.read_text()).get("features") == 15 for path in backup.rglob("schema.json"))
    for split, count in [("train", 5), ("val", 3), ("test", 2), ("all", 10)]:
        dataset = FKDDataset(root, cfg, split)
        assert len(dataset) == count
        for value in range(len(dataset)):
            location = dataset.locate_window(value)
            file = Path(location["path"])
            item = next(entry for entry in bins.values() if entry["new"] == file)
            expected = np.frombuffer(item["bytes"], dtype=np.float32).reshape(item["shape"])
            np.testing.assert_array_equal(dataset.load_window(value), expected[location["window_index"]])
            assert location["split"] == item["split"]
    assert len(FKDDataset(root / "sim" / "road" / "valid", cfg, "val")) == 2
    snapshot = tree_contents(root)
    again = migration.reorganize(root, apply=True)
    assert again["status"] == "already-flat"
    assert tree_contents(root) == snapshot


@pytest.mark.parametrize("held", ["builder", "writer"])
def test_active_locks_reject_before_any_move(migration, tmp_path, held):
    root, _, documents, bins, _ = old_dataset(tmp_path)
    lock_path = root / ".builder.lock" if held == "builder" else next(iter(documents)) / ".writer.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = tree_contents(tmp_path)
        with pytest.raises(RuntimeError, match="(?i)active|lock|writer|builder"):
            migration.reorganize(root, apply=True)
        assert tree_contents(tmp_path) == original
        assert all(path.exists() and file_identity(path) == item["identity"] for path, item in bins.items())


@pytest.mark.parametrize("fail_at", [2, 8, 12])
def test_mid_move_failure_restores_paths_json_and_bin_identities(migration, tmp_path, monkeypatch, fail_at):
    root, _, _, bins, _ = old_dataset(tmp_path)
    original = tree_contents(root)
    rename = migration.os.rename
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("injected migration rename failure")
        return rename(*args, **kwargs)

    monkeypatch.setattr(migration.os, "rename", fail_once)
    with pytest.raises(OSError, match="injected migration rename failure"):
        migration.reorganize(root, apply=True)
    assert calls > fail_at  # At least one inverse rename must have run.
    assert tree_contents(root) == original
    assert all(file_identity(path) == item["identity"] for path, item in bins.items())


def test_untracked_bins_reject_without_mutation(migration, tmp_path):
    root, _, documents, _, _ = old_dataset(tmp_path)
    group = next(iter(documents))
    np.zeros((1, 5, 42), dtype=np.float32).tofile(group / "unexpected.bin")
    original = tree_contents(tmp_path)
    with pytest.raises(ValueError, match="(?i)unexpected|untracked|bin"):
        migration.reorganize(root, apply=True)
    assert tree_contents(tmp_path) == original


@pytest.mark.parametrize("missing", ["dataset_build.json", ".builder.lock"])
def test_missing_build_identity_or_lock_is_not_recreated(migration, tmp_path, missing):
    root, _, _, _, _ = old_dataset(tmp_path)
    (root / missing).unlink()
    original = tree_contents(tmp_path)
    with pytest.raises(ValueError):
        migration.reorganize(root, apply=False)
    assert tree_contents(tmp_path) == original
