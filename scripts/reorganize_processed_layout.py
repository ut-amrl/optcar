#!/usr/bin/env python3
"""Move an existing builder output to source/terrain/split without rebuilding.

Dry-run by default. Only directory renames and metadata updates are performed;
binary files are never opened for writing. Requires a stopped dataset builder.
"""
import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import tempfile
import uuid


LAYOUT = "source_terrain_split_v1"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def _atomic_bytes(path, content):
    fd, name = tempfile.mkstemp(prefix=".layout-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def _lock(stack, path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular lock file: {path}")
    stream = stack.enter_context(path.open("rb"))
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another builder or writer is active: {path}") from error


def _identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def _inspect(root, index):
    flat = index.get("output_layout") == LAYOUT
    if index.get("output_layout") not in (None, LAYOUT):
        raise ValueError("Unrecognized output layout")
    groups, seen = [], set()
    totals = {"bins": 0, "windows": 0, "bytes": 0}
    manifests = sorted(root.glob("*/*/manifest.json"))
    if not manifests or set(root.rglob("manifest.json")) != set(manifests):
        raise ValueError("Expected only builder group manifests two folders below the root")
    for path in manifests:
        old = path.parent
        original = path.read_bytes()
        doc = json.loads(original)
        source, terrain = doc["source"], doc["terrain"]
        if source not in ("real", "sim") or not terrain or Path(terrain).name != terrain or terrain in (".", ".."):
            raise ValueError(f"Invalid source or terrain: {path}")
        expected = root / source / terrain if flat else root / terrain / source
        if old != expected or bool(doc.get("flat_processed", False)) != flat or not doc.get("process_windows", True):
            raise ValueError(f"Incompatible group layout: {path}")
        if doc.get("schema", {}).get("dtype") != "float32" or doc["schema"].get("features") != 42:
            raise ValueError(f"Expected float32 processed states with 42 features: {path}")
        new = root / source / terrain
        if not flat and (new.exists() or new.is_symlink()):
            raise ValueError(f"Destination already exists: {new}")
        if any(p.is_symlink() for p in old.rglob("*")) or old.is_symlink():
            raise ValueError(f"Symlinks are unsupported: {old}")
        if not flat:
            if {p.name for p in old.iterdir()} != {"manifest.json", ".writer.lock", "metadata", "processed"}:
                raise ValueError(f"Unexpected files in group: {old}")
            if {p.name for p in (old / "processed").iterdir()} != {"data", "metadata"}:
                raise ValueError(f"Unexpected processed folders: {old}")
            if not {p.name for p in (old / "processed/data").iterdir()} <= set(SPLITS):
                raise ValueError(f"Unexpected split folders: {old}")
        metadata = old / ("metadata" if flat else "processed/metadata")
        for name in ("dataset_config.py", "schema.json"):
            if not (metadata / name).is_file():
                raise ValueError(f"Missing processed metadata: {metadata / name}")
        if json.loads((metadata / "schema.json").read_text()).get("features") != 42:
            raise ValueError(f"Wrong processed metadata: {metadata}")
        updated = copy.deepcopy(doc)
        records, mapping = [], {}
        shards = list(updated.get("processed_shards", []))
        shards += [s for e in updated["episodes"].values() for s in e.get("shards", [])]
        for shard in shards:
            split = shard["split"]
            if split not in SPLITS:
                raise ValueError(f"Invalid split in {path}")
            relative = shard["path"]
            parts = PurePosixPath(relative).parts
            prefix = (SPLITS[split],) if flat else ("processed", "data", split)
            if parts[:-1] != prefix or not parts[-1].endswith(".bin"):
                raise ValueError(f"Unexpected bin path: {relative}")
            target = str(PurePosixPath(SPLITS[split], parts[-1]))
            binary = old / relative
            shape = shard["shape"]
            if len(shape) != 3 or shape[-1] != 42 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f"Invalid processed shape: {binary}")
            identity = _identity(binary)
            if identity[0] != root.stat().st_dev or identity[2] != math.prod(shape) * 4:
                raise ValueError(f"Bin size or filesystem mismatch: {binary}")
            if binary in seen or target in mapping.values():
                raise ValueError(f"Duplicate processed bin: {binary}")
            seen.add(binary)
            mapping[relative] = target
            records.append({"old": relative, "new": target, "identity": identity, "shape": shape, "split": split})
            shard["path"] = target
            totals["bins"] += 1
            totals["windows"] += shape[0]
            totals["bytes"] += identity[2]
        if set(old.rglob("*.bin")) != {old / p for p in mapping}:
            raise ValueError(f"Unlisted binary files in {old}; refusing to move them")
        declared = sum(e["windows"] for e in doc["episodes"].values())
        if declared != sum(r["shape"][0] for r in records):
            raise ValueError(f"Incomplete windows in {old}")
        if any(e.get("stored_windows", e["windows"]) != e["windows"] for e in doc["episodes"].values()):
            raise ValueError(f"Incomplete recording in {old}")
        updated["flat_processed"] = True
        groups.append({"old": old, "new": new, "original": original, "updated": updated, "records": records})
    if set(root.rglob("*.bin")) != seen:
        raise ValueError("Unlisted bins outside the declared dataset groups")
    return flat, groups, totals


def reorganize(root, apply=False):
    """Preflight or migrate stopped legacy builder output, preserving every bin."""
    root = Path(root).expanduser().resolve(strict=True)
    index_path = root / "dataset_build.json"
    with ExitStack() as stack:
        _lock(stack, root / ".builder.lock")
        for manifest in sorted(root.glob("*/*/manifest.json")):
            _lock(stack, manifest.parent / ".writer.lock")
        if index_path.is_symlink() or not index_path.is_file():
            raise ValueError(f"Missing regular build identity: {index_path}")
        original_index = index_path.read_bytes()
        index = json.loads(original_index)
        flat, groups, totals = _inspect(root, index)
        result = {"status": "already-flat" if flat else "planned", "root": str(root), **totals,
                  "groups": [str(g["new"]) for g in groups]}
        if flat or not apply:
            return result
        backup = root.parent / f".{root.name}-layout-backup-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        backup.mkdir()
        if backup.stat().st_dev != root.stat().st_dev:
            backup.rmdir()
            raise ValueError("Backup and dataset must be on the same filesystem")
        _atomic_bytes(backup / "dataset_build.json.original", original_index)
        for group in groups:
            destination = backup / group["new"].relative_to(root)
            destination.mkdir(parents=True)
            _atomic_bytes(destination / "manifest.json.original", group["original"])
        journal = {"root": str(root), "status": "moving", "renames": [], "bins": [
            {"group": str(g["new"]), **record} for g in groups for record in g["records"]]}
        created, changed_manifests = [], []
        journal_path = backup / "migration.json"

        def save_journal():
            _atomic_bytes(journal_path, _json_bytes(journal))

        def mkdir(path):
            if not path.exists():
                path.mkdir()
                created.append(path)

        def rename(source, destination):
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"Destination already exists: {destination}")
            entry = {"from": str(source), "to": str(destination), "done": False}
            journal["renames"].append(entry)
            save_journal()
            os.rename(source, destination)
            entry["done"] = True
            _sync_directory(source.parent)
            _sync_directory(destination.parent)
            save_journal()

        save_journal()
        try:
            for group in groups:
                new = group["new"]
                mkdir(new.parent)
                rename(group["old"], new)
                rename(new / "metadata", backup / new.relative_to(root) / "raw_metadata")
                for split, folder in SPLITS.items():
                    source = new / "processed/data" / split
                    if source.exists():
                        rename(source, new / folder)
                    else:
                        mkdir(new / folder)
                rename(new / "processed/metadata", new / "metadata")
                changed_manifests.append(group)
                _atomic_bytes(new / "manifest.json", _json_bytes(group["updated"]))
            for group in groups:
                for record in group["records"]:
                    if _identity(group["new"] / record["new"]) != record["identity"]:
                        raise RuntimeError("A binary changed during migration")
            index["output_layout"] = LAYOUT
            _atomic_bytes(index_path, _json_bytes(index))
            _, checked, actual = _inspect(root, index)
            if actual != totals or len(checked) != len(groups):
                raise RuntimeError("Dataset totals changed during migration")
            journal["status"] = "complete"
            save_journal()
        except BaseException as failure:
            rollback_errors = []

            def attempt(operation):
                try:
                    operation()
                except BaseException as error:
                    rollback_errors.append(str(error))

            # Move prewritten originals back: recovery must not allocate new
            # metadata files if a full disk caused the original failure.
            attempt(lambda: os.replace(backup / "dataset_build.json.original", index_path))
            for group in changed_manifests:
                attempt(lambda g=group: os.replace(backup / g["new"].relative_to(root) / "manifest.json.original",
                                                   g["new"] / "manifest.json"))

            def remove_empty(path):
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()

            # Always attempt every inverse move, even if metadata recovery or
            # syncing fails. Unknown/nonempty directories are never removed.
            for path in reversed(created):
                attempt(lambda p=path: remove_empty(p))
            for entry in reversed(journal["renames"]):
                source, destination = Path(entry["from"]), Path(entry["to"])
                if destination.exists() and not source.exists():
                    attempt(lambda s=source, d=destination: os.rename(d, s))
                    attempt(lambda p=source.parent: _sync_directory(p))
                    attempt(lambda p=destination.parent: _sync_directory(p))
            for path in reversed(created):
                attempt(lambda p=path: remove_empty(p))
            journal["status"] = "rollback-incomplete" if rollback_errors else "rolled-back"
            journal["rollback_errors"] = rollback_errors
            attempt(save_journal)
            if rollback_errors:
                raise RuntimeError(f"Migration failed; inspect recovery journal in {backup}: {rollback_errors}") from failure
            raise
        # The transaction is complete; cleanup never removes nonempty folders.
        for group in groups:
            for path in (group["new"] / "processed/data", group["new"] / "processed", group["old"].parent):
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()
        return {**result, "status": "applied", "backup_root": str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Existing processed output root")
    parser.add_argument("--apply", action="store_true", help="Apply renames after validating all bins")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"Migration interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(reorganize(args.root, args.apply), indent=2))


if __name__ == "__main__":
    main()
