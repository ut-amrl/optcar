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
from optcar.utils.file_io import read_config
from optcar.utils.math import yaw_from_quat_torch


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


class DatasetViewer:
    """Bounded datasets/memmaps, with one copied window returned per request."""
    def __init__(self):
        self.datasets = OrderedDict()
        self.lock = threading.RLock()

    def open(self, roots, config_path, split="train"):
        if isinstance(roots, str):
            roots = [line.strip() for line in roots.splitlines() if line.strip()]
        if not isinstance(roots, list) or not roots or not all(isinstance(root, str) and root.strip() for root in roots):
            raise ValueError("Enter at least one dataset directory, one per line")
        roots = list(dict.fromkeys(str(Path(root.strip()).expanduser().resolve()) for root in roots))
        for root in roots:
            if not Path(root).is_dir():
                raise ValueError(f"Dataset directory does not exist: {root}")
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("Enter the dataset config path")
        config_path = str(Path(config_path.strip()).expanduser().resolve())
        if split not in ("train", "val", "test", "all"):
            raise ValueError("Choose train, val, test or all splits")
        config = read_dataset_config(config_path)
        dataset = FKDDataset(roots, config, split=split, allow_wheeledlab=True)
        identifier = uuid.uuid4().hex
        with self.lock:
            self.datasets[identifier] = dataset
            while len(self.datasets) > 4:
                self.datasets.popitem(last=False)
        return {"dataset_id": identifier, "roots": roots, "config_path": config_path,
                "split": split, "count": len(dataset), "files": len(dataset.shards),
                "samples": config.window_steps, "features": 42, **asdict(config)}

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
            location = dataset.locate_window(index)
            sample = dataset.get_fkd_sample(index)
        data = sample["full_data"]
        if not torch.isfinite(data).all():
            raise ValueError(f"Trajectory {index} contains non-finite values in {location['path']}")
        current = dataset.config.history_steps - 1
        result = {"index": index, **location, "car_number": int(sample["car_number"]),
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
