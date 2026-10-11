"""Check shared-shard recovery without touching user datasets or jobs."""
import json

import numpy as np
import pytest
import torch

from optcar.dataset import DataConfig, DatasetWriter, FKDDataset


def packed_fixture(count, start=0):
    values = torch.zeros((count, 5, 42), dtype=torch.float32)
    values[:, :, 0] = torch.arange(start, start + count)[:, None]
    values[:, :, (3, 16, 29)] = 1
    return values


def write_fixture(writer, name, values, split, raw_path):
    return writer.write_window_batches(name, [values], split, name, raw_path, [1, 8, 15])


def read_markers(root, manifest, split):
    values = []
    for shard in manifest.get('processed_shards', []):
        if shard['split'] == split:
            rows = np.fromfile(root / shard['path'], dtype=np.float32).reshape(shard['shape'])
            values.extend(rows[:, 0, 0].tolist())
    return values


def test_successful_split_flush_is_durable_when_next_split_fails(tmp_path, monkeypatch):
    cfg = DataConfig(history_steps=3, future_steps=2, shard_windows=5)
    root = tmp_path / 'assembled'
    train, val = packed_fixture(2), packed_fixture(3, 100)
    with DatasetWriter(root, cfg, 'sim', 'road') as writer:
        write_fixture(writer, 'train_input', train, 'train', tmp_path / 'train.bin')
        write_fixture(writer, 'val_input', val, 'val', tmp_path / 'val.bin')
        original_binary = writer._binary

        def fail_val(filename, values):
            if '/val/' in filename:
                raise OSError('fixture validation write failure')
            return original_binary(filename, values)

        monkeypatch.setattr(writer, '_binary', fail_val)
        with pytest.raises(OSError, match='validation write failure'):
            writer.flush_window_batches()

        checkpoint = json.loads((root / 'manifest.json').read_text())
        assert [shard['split'] for shard in checkpoint['processed_shards']] == ['train']
        assert checkpoint['episodes']['train_input']['stored_windows'] == 2
        assert checkpoint['episodes']['val_input']['stored_windows'] == 0
        assert read_markers(root, checkpoint, 'train') == [0, 1]
        assert not list((root / 'processed/data/val').glob('*.bin'))

        monkeypatch.setattr(writer, '_binary', original_binary)
        writer.flush_window_batches()

    before = {path: path.read_bytes() for path in root.rglob('*.bin')}
    with DatasetWriter(root, cfg, 'sim', 'road') as writer:
        assert writer.contains('train_input') and writer.contains('val_input')
        assert write_fixture(writer, 'train_input', train, 'train', tmp_path / 'train.bin') == 0
        assert write_fixture(writer, 'val_input', val, 'val', tmp_path / 'val.bin') == 0
    assert before == {path: path.read_bytes() for path in root.rglob('*.bin')}
    final = json.loads((root / 'manifest.json').read_text())
    assert len(final['processed_shards']) == 2
    assert read_markers(root, final, 'val') == [100, 101, 102]
    assert len(FKDDataset(root, cfg, 'train')) == 2
    assert len(FKDDataset(root, cfg, 'val')) == 3


def test_committed_prefix_is_skipped_when_unsaved_carryover_is_lost(tmp_path):
    cfg = DataConfig(history_steps=3, future_steps=2, shard_windows=5)
    root = tmp_path / 'assembled'
    first, second = packed_fixture(3), packed_fixture(4, 100)
    writer = DatasetWriter(root, cfg, 'sim', 'road')
    try:
        write_fixture(writer, 'first', first, 'train', tmp_path / 'first.bin')
        write_fixture(writer, 'second', second, 'train', tmp_path / 'second.bin')
        writer.commit()
        checkpoint = json.loads((root / 'manifest.json').read_text())
        assert checkpoint['episodes']['first']['stored_windows'] == 3
        assert checkpoint['episodes']['second']['stored_windows'] == 2
        assert read_markers(root, checkpoint, 'train') == [0, 1, 2, 100, 101]
    finally:
        # Simulate losing only in-memory carryover after a completed checkpoint.
        writer.lock.close()

    with DatasetWriter(root, cfg, 'sim', 'road') as resumed:
        assert write_fixture(resumed, 'first', first, 'train', tmp_path / 'first.bin') == 0
        assert write_fixture(resumed, 'second', second, 'train', tmp_path / 'second.bin') == 2
    final = json.loads((root / 'manifest.json').read_text())
    assert [shard['shape'][0] for shard in final['processed_shards']] == [5, 2]
    assert read_markers(root, final, 'train') == [0, 1, 2, 100, 101, 102, 103]
    assert final['episodes']['second']['stored_windows'] == 4
    assert len(FKDDataset(root, cfg, 'train')) == 7
