"""Configuration and atomic artifact writes."""
import json
import os
import re
import tempfile
from pathlib import Path

import yaml


def read_config(path):
    with Path(path).expanduser().open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_yaml(path, value):
    atomic_text(path, yaml.safe_dump(value, sort_keys=False))


def safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)):
        raise ValueError("Names must use letters, numbers, underscores, dots, or hyphens")
    return str(name)


def terrain_path(root, terrain):
    return Path(root).expanduser().resolve() / "syn_data" / "terrains" / safe_name(terrain)
