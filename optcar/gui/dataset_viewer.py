"""Processed-window inspection through the same FKD dataset used for training.

Sample slicing follows WheeledLab's FKD loader. Config files are read as data;
loading its saved Python metadata does not import WheeledLab or execute Python.
"""
import ast
from collections import OrderedDict
from dataclasses import asdict
import operator
from pathlib import Path
import threading
import uuid

import torch
import yaml

from optcar.dataset import DataConfig, FKDDataset
from optcar.utils.file_io import read_config, write_json, write_yaml
from optcar.utils.math import yaw_from_quat_torch
from optcar.gui.dataloader_setup import load_python_setup, validate_loader_config


def _metadata_values(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ProcessedDatasetConfig"]
    if len(classes) != 1:
        raise ValueError("Choose the processed metadata/dataset_config.py containing ProcessedDatasetConfig")
    values = {}
    # WheeledLab's saved metadata contains literal class attributes. Never import
    # the file: its dataloader/training configs may depend on Isaac or ROS.
    for node in classes[0].body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                try:
                    values[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
    return values


def read_dataset_config(path):
    """Accept OptCar YAML/JSON or WheeledLab's saved processed metadata."""
    path = Path(path).expanduser().resolve()
    try:
        if path.suffix.lower() == ".py":
            values = _metadata_values(path)
        elif path.suffix.lower() in (".yaml", ".yml", ".json"):
            doc = read_config(path)
            values = doc.get("data", doc.get("config", doc))
            if not isinstance(values, dict):
                raise ValueError("Dataset configuration must be a mapping")
        else:
            raise ValueError("Choose a dataset YAML, JSON or processed dataset_config.py")
    except (SyntaxError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read dataset config {path}: {exc}") from exc

    return _parse_dataset_config(values)


def _parse_dataset_config(values):
    for key in ("features", "feature_dim"):
        if key in values and values[key] != 42:
            raise ValueError("Processed FKD datasets require 42 features (including vehicle ID)")
    for key in ("dtype", "data_dtype"):
        if key in values and values[key] != "float32":
            raise ValueError("Processed FKD datasets require float32 data")
    if values.get("quaternion_order", "wxyz") != "wxyz":
        raise ValueError("Processed FKD quaternions must use wxyz order")

    history = values.get("history_steps", values.get("history_horizon", values.get("history_horizon_steps")))
    future = values.get("future_steps")
    if future is not None and not isinstance(future, (int, str)):
        raise ValueError("Future transitions must be an integer")
    if not isinstance(future, int):
        horizon = values.get("future_horizon", values.get("future_horizon_steps"))
        future = horizon - 1 if isinstance(horizon, int) else None
    dt = values.get("dt", values.get("control_dt"))
    if dt is None and "sim_dt" in values:
        dt = values["sim_dt"] * values.get("decimation", 1)
    if history is None or future is None or dt is None:
        raise ValueError("Config must declare history, future and timestep; use the saved processed dataset_config.py or your OptCar data YAML")
    if isinstance(history, bool) or isinstance(future, bool) or isinstance(dt, bool):
        raise ValueError("History/future must be integers and timestep must be a positive number")
    settings = {"history_steps": history, "future_steps": future, "dt": dt}
    settings.update({key: values[key] for key in ("stride", "shard_windows") if key in values})
    config = DataConfig(**settings)
    for key in ("total_horizon", "chunk_horizon", "window_steps"):
        if key in values and values[key] != config.window_steps:
            raise ValueError(f"{key} differs from history + future transitions")
    return config


def _viewer_config_path(path, saving=False):
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Enter a dataset config path including the filename")
    requested = Path(str(path).strip()).expanduser()
    path = requested.resolve()
    if saving:
        protected = {"manifest.json", "dataset_build.json", "schema.json"}
        if requested.name.lower() in protected or path.name.lower() in protected or path.suffix.lower() == ".py":
            raise ValueError("Generated dataset metadata cannot be overwritten; choose a new config filename")
        if path.suffix.lower() not in (".yaml", ".yml", ".json") or path.is_dir():
            raise ValueError("Choose a config filename ending in .yaml, .yml or .json")
    return path


def _viewer_roots(roots, base=None, allow_empty=False):
    if isinstance(roots, str):
        roots = [line.strip() for line in roots.splitlines() if line.strip()]
    if allow_empty and (roots is None or roots == []):
        return []
    if not isinstance(roots, list) or not roots or not all(isinstance(root, str) and root.strip() for root in roots):
        raise ValueError("Enter at least one dataset directory, one per line")
    normalized = []
    for root in roots:
        path = Path(root.strip()).expanduser()
        if base is not None and not path.is_absolute():
            path = base / path
        normalized.append(str(path.resolve()))
    return list(dict.fromkeys(normalized))


def _viewer_split(split):
    if split not in ("train", "val", "test", "all"):
        raise ValueError("Choose train, val, test or all splits")
    return split


def _editable_data_config(data):
    if isinstance(data, DataConfig):
        data = asdict(data)
    if not isinstance(data, dict) or any(key not in data for key in ("history_steps", "future_steps", "dt")):
        raise ValueError("Enter history, future and timestep for the dataset")
    if any(isinstance(data[key], bool) for key in ("history_steps", "future_steps", "dt")) or any(
            not isinstance(data[key], int) for key in ("history_steps", "future_steps")):
        raise ValueError("History/future must be integers and timestep must be a positive number")
    try:
        return _parse_dataset_config(data)
    except TypeError as exc:
        raise ValueError("History/future must be integers and timestep must be a positive number") from exc


def _metadata_roots(path, doc):
    if path.name == "dataset_build.json" and isinstance(doc.get("config"), dict):
        return [str(path.parent)]
    if path.name == "manifest.json" and doc.get("schema", {}).get("features") == 42 and "episodes" in doc:
        return [str(path.parent)]
    if path.parent.name == "metadata" and path.name in ("dataset_config.py", "schema.json"):
        candidate = path.parent.parent
        for parent in (candidate, *candidate.parents):
            manifest = parent / "manifest.json"
            if manifest.is_file():
                saved = read_config(manifest)
                if saved.get("schema", {}).get("features") == 42 and "episodes" in saved:
                    return [str(parent)]
        if (candidate / "data").is_dir() or any(candidate.glob("*.bin")):
            return [str(candidate)]
    return []


def load_viewer_config(path):
    """Load editable dimensions and ordered dataset paths without reading bins."""
    path = _viewer_config_path(path)
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"Cannot read dataset config {path}: {exc}") from exc
        if any(isinstance(node, ast.FunctionDef) and node.name in ("dataloader_config", "get_dataloader_config", "config") for node in tree.body):
            return load_python_setup(path)
    config = read_dataset_config(path)
    doc = {} if path.suffix.lower() == ".py" else read_config(path)
    if "datasets" in doc:
        roots = doc["datasets"]
    elif isinstance(doc.get("training"), dict) and "datasets" in doc["training"]:
        roots = doc["training"]["datasets"]
    else:
        roots = _metadata_roots(path, doc)
    return {"config_path": str(path),
            "roots": _viewer_roots(roots, path.parent, allow_empty=True),
            "data": asdict(config), "split": _viewer_split(doc.get("split", "train"))}


def save_viewer_config(path, roots, data, split):
    """Save a reusable viewer config, preserving other existing config sections."""
    path = _viewer_config_path(path, saving=True)
    roots = _viewer_roots(roots)
    config = _editable_data_config(data)
    split = _viewer_split(split)
    try:
        doc = read_config(path) if path.exists() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot read dataset config {path}: {exc}") from exc
    if "training" in doc and not isinstance(doc["training"], dict):
        raise ValueError("Existing training configuration must be a mapping")
    values = doc.get("data", {})
    if not isinstance(values, dict):
        raise ValueError("Existing dataset configuration must be a mapping")
    values = values.copy()
    values.update({key: getattr(config, key) for key in ("history_steps", "future_steps", "dt")})
    config = _parse_dataset_config(values)
    doc.update({"datasets": roots,
                "data": values,
                "split": split})
    if isinstance(doc.get("training"), dict) and "datasets" in doc["training"]:
        doc["training"]["datasets"] = roots
    if path.suffix.lower() == ".json":
        write_json(path, doc)
    else:
        write_yaml(path, doc)
    return {"config_path": str(path), "roots": roots,
            "data": asdict(config), "split": split}


class DatasetViewer:
    """Bounded datasets/memmaps, with one copied window returned per request."""
    def __init__(self):
        self.datasets = OrderedDict()
        self.orders = {}
        self.lock = threading.RLock()

    def open(self, roots=None, config_path=None, split=None, data=None, dl_config=None, is_test_set=False):
        if data is None:
            loaded = load_viewer_config(config_path)
            if loaded["data"] is None:
                raise ValueError("This loader config has no saved horizons; enter history and future from the model config")
            config = DataConfig(**loaded["data"])
            if dl_config is None:
                dl_config = loaded.get("dl_config")
            if roots is None or roots == [] or (isinstance(roots, str) and not roots.strip()):
                roots = loaded["roots"]
            config_path = loaded["config_path"]
            split = loaded["split"] if split is None else split
        else:
            config = _editable_data_config(data)
            config_path = str(_viewer_config_path(config_path)) if config_path else None
        if not isinstance(is_test_set, bool):
            raise ValueError("is_test_set must be true or false")
        loader = validate_loader_config(dl_config) if dl_config is not None else None
        if loader is not None:
            roots = loader.test_dataset_path if is_test_set else loader.dataset_path
            if loader.sim_dt != config.dt:
                raise ValueError("Dataloader sim_dt differs from the viewer timestep")
        roots = _viewer_roots(roots)
        for root in roots:
            if not Path(root).is_dir():
                raise ValueError(f"Dataset directory does not exist: {root}")
        split = _viewer_split("train" if split is None else split)
        dataset = FKDDataset(roots, config, split=split, allow_wheeledlab=True,
                             dl_config=loader, is_test_set=is_test_set)
        identifier = uuid.uuid4().hex
        with self.lock:
            self.datasets[identifier] = dataset
            while len(self.datasets) > 4:
                expired, _ = self.datasets.popitem(last=False)
                self.orders.pop(expired, None)
        return {"dataset_id": identifier, "roots": roots, "config_path": config_path,
                "split": split, "count": len(dataset), "files": len(dataset.shards),
                "samples": config.window_steps, "features": 42, "selection": dataset.selection_summary,
                "is_test_set": is_test_set, "shuffled": False, **asdict(config)}

    def shuffle(self, dataset_id, seed=None):
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("Shuffle seed must be a nonnegative integer or empty")
        with self.lock:
            dataset = self.datasets.get(dataset_id)
            if dataset is None:
                raise ValueError("Dataset expired; load the dataset again")
            if seed is None:
                seed = torch.randint(0, 2 ** 31, ()).item()
            generator = torch.Generator().manual_seed(seed)
            self.orders[dataset_id] = torch.randperm(len(dataset), generator=generator)
            self.datasets.move_to_end(dataset_id)
            return {"dataset_id": dataset_id, "seed": seed, "count": len(dataset), "shuffled": True}

    def sample(self, dataset_id, index):
        try:
            if isinstance(index, bool):
                raise TypeError
            index = operator.index(index)
        except TypeError:
            raise ValueError("Trajectory index must be an integer") from None
        with self.lock:
            dataset = self.datasets.get(dataset_id)
            if dataset is None:
                raise ValueError("Dataset expired; load the dataset again")
            if not 0 <= index < len(dataset):
                raise ValueError(f"Trajectory index must be between 0 and {len(dataset) - 1}")
            self.datasets.move_to_end(dataset_id)
            order = self.orders.get(dataset_id)
            dataset_index = int(order[index]) if order is not None else index
            location = dataset.locate_window(dataset_index)
            sample = dataset.get_fkd_sample(dataset_index)
        data = sample["full_data"]
        if not torch.isfinite(data).all():
            raise ValueError(f"Trajectory {index} contains non-finite values in {location['path']}")
        current = dataset.config.history_steps - 1
        result = {"index": index, "dataset_index": dataset_index, "shuffled": order is not None,
                  **location, "car_number": int(sample["car_number"]),
                  "current_step": current, "dt": dataset.config.dt,
                  "times": ((torch.arange(len(data), dtype=torch.float64) - current) * dataset.config.dt).tolist(),
                  "world": sample["full_wf_states"].tolist(),
                  "body": sample["full_bf_states"].tolist(),
                  "delta": data[:, 26:39].tolist(), "actions": data[:, 39:41].tolist()}
        for frame in ("world", "body"):
            states = sample["full_wf_states" if frame == "world" else "full_bf_states"]
            result[frame + "_yaw"] = yaw_from_quat_torch(states[:, 3:7]).squeeze(-1).tolist()
        for key in ("history", "future", "history_actions", "future_actions", "delta_history", "delta_future"):
            result[key] = sample[key].tolist()
        return result
