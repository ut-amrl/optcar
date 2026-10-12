"""Edit Python DataloaderConfig factories without executing them."""
import ast
from dataclasses import asdict, fields
import math
from pathlib import Path

import torch

from optcar.dataloader_config import DataloaderConfig
from optcar.utils.file_io import atomic_text


def config_values(config):
    values = asdict(config)
    values["dtype"] = str(config.dtype).removeprefix("torch.")
    return values


def default_loader_values():
    return config_values(DataloaderConfig(dataset_path=[], test_dataset_path=[],
                                         num_runs=-1, test_num_runs=-1, device="cpu"))


def validate_loader_config(values, base=None):
    if not isinstance(values, dict):
        raise ValueError("Dataloader configuration must be a mapping")
    known = {field.name for field in fields(DataloaderConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown DataloaderConfig fields: {', '.join(sorted(unknown))}")
    values = {**default_loader_values(), **values}
    for prefix in ("", "test_"):
        paths = values[prefix + "dataset_path"]
        if isinstance(paths, str):
            paths = [paths]
        if paths is None:
            paths = []
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError(f"{prefix}dataset_path must contain folder paths")
        normalized = []
        for path in paths:
            path = Path(path.strip()).expanduser()
            if base is not None and not path.is_absolute():
                path = Path(base) / path
            normalized.append(str(path.resolve()))
        values[prefix + "dataset_path"] = normalized
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"List each folder only once in {prefix}dataset_path")
        for name, nullable in (("num_runs", False), ("num_trajectories", True)):
            key = prefix + name
            limits = values[key]
            if limits is None and nullable:
                continue
            entries = limits if isinstance(limits, list) else [limits] * len(paths)
            if isinstance(limits, list) and len(entries) != len(paths):
                raise ValueError(f"{key} must match the number of paths")
            if not isinstance(limits, (list, int)) or isinstance(limits, bool) or any(
                    not (nullable and item is None) and
                    (isinstance(item, bool) or not isinstance(item, int) or item < -1) for item in entries):
                raise ValueError(f"{key} must be integers >= 0 or -1 for all")
    seed = values["file_selection_seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("Selection seed must be a nonnegative integer or empty")
    dt = values["sim_dt"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("sim_dt must be a positive timestep")
    dtype = values["dtype"]
    if dtype != "float32" and dtype != torch.float32:
        raise ValueError("The standard processed dataset uses float32")
    values["dtype"] = torch.float32
    if not isinstance(values["device"], str):
        raise ValueError("Device must be a torch device name")
    try:
        torch.device(values["device"])
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Device must be cpu or a valid torch device name") from exc
    for field in fields(DataloaderConfig):
        value = values[field.name]
        if field.type is bool and not isinstance(value, bool):
            raise ValueError(f"{field.name} must be true or false")
        if field.name.startswith("plot_") or field.name == "check_val_every_n_epoch" or field.name == "batch_size":
            if field.name == "batch_size" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")
    return DataloaderConfig(**values)


def _literal(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "float32":
        return "float32"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "device" and len(node.args) == 1 and not node.keywords:
        return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


def read_python_loader(path):
    """Read literal factory keyword arguments; never execute config imports/code."""
    path = Path(path).expanduser().resolve()
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Cannot read dataloader config: {exc}") from exc
    factories = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("dataloader_config", "get_dataloader_config", "config")]
    if not factories:
        raise ValueError("Choose a Python file defining dataloader_config()")
    returns = [node for node in factories[0].body if isinstance(node, ast.Return)]
    call = returns[0].value if len(returns) == 1 else None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "DataloaderConfig" or call.args:
        raise ValueError("Factory must return DataloaderConfig with explicit keyword arguments")
    values = {}
    try:
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in values:
                raise ValueError("Expanded or duplicate configuration arguments are unsupported")
            values[keyword.arg] = _literal(keyword.value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Config fields must be literal values (paths, numbers, lists, booleans or torch.float32)") from exc
    return validate_loader_config(values, path.parent), tree


def infer_viewer_data(config):
    """Horizons remain outside DataloaderConfig."""
    from optcar.gui.dataset_viewer import read_dataset_config
    for root in [*config.dataset_path, *config.test_dataset_path]:
        path = Path(root)
        for parent in (path, *path.parents):
            for candidate in (parent / "manifest.json", parent / "metadata" / "dataset_config.py", parent / "metadata" / "schema.json"):
                if candidate.is_file():
                    data = read_dataset_config(candidate)
                    if data.dt != config.sim_dt:
                        raise ValueError(f"{candidate}: timestep differs from sim_dt")
                    return asdict(data)
    return None


def load_python_setup(path):
    path = Path(path).expanduser().resolve()
    config, tree = read_python_loader(path)
    data = infer_viewer_data(config)
    # Optional viewer-only horizons are a separate literal, not loader fields.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "viewer_data" for target in node.targets):
            from optcar.gui.dataset_viewer import _editable_data_config
            data = asdict(_editable_data_config(ast.literal_eval(node.value)))
            if data["dt"] != config.sim_dt:
                raise ValueError("Viewer timestep differs from sim_dt")
    return {"config_path": str(path), "dl_config": config_values(config),
            "roots": config.dataset_path, "data": data, "split": "all", "is_test_set": False}


def save_python_setup(path, values, data=None):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Choose a full config path and .py filename")
    requested = Path(path.strip()).expanduser()
    path = requested.resolve()
    if not requested.is_absolute() or path.suffix.lower() != ".py" or path.is_dir():
        raise ValueError("Choose a full config path ending in .py")
    if requested.name == "dataset_config.py" or path.name == "dataset_config.py":
        raise ValueError("Generated dataset metadata is read-only; choose a different filename")
    config = validate_loader_config(values)
    if not config.dataset_path:
        raise ValueError("Add at least one dataset folder before saving")
    if path.exists():
        read_python_loader(path)  # Never overwrite unrelated Python source/metadata.
    lines = ["# DataloaderConfig factory for OptCar.", "import torch",
             "from optcar.dataloader_config import DataloaderConfig", "",
             "", "def dataloader_config() -> DataloaderConfig:", "    return DataloaderConfig("]
    for key, value in config_values(config).items():
        rendered = "torch.float32" if key == "dtype" else repr(value)
        lines.append(f"        {key}={rendered},")
    lines.extend(["    )", ""])
    if data is not None:
        from optcar.gui.dataset_viewer import _editable_data_config
        data = asdict(_editable_data_config(data))
        if data["dt"] != config.sim_dt:
            raise ValueError("Viewer timestep differs from sim_dt")
        lines.extend(["", "# Viewer horizons; training reads these from its model config.",
                      f"viewer_data = {data!r}", ""])
    atomic_text(path, "\n".join(lines))
    return {"config_path": str(path), "dl_config": config_values(config),
            "roots": config.dataset_path, "data": data, "split": "all", "is_test_set": False}


def browse_processed_folder(path, data):
    from optcar.gui.dataset_viewer import _editable_data_config
    config = _editable_data_config(data)
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Folder does not exist: {root}")
    folders, bins = [], []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if entry.is_dir():
            folders.append({"name": entry.name, "path": str(entry)})
        elif entry.suffix == ".bin" and entry.is_file():
            count, remainder = divmod(entry.stat().st_size, config.window_steps * 42 * 4)
            bins.append({"name": entry.name, "path": str(entry), "trajectories": count,
                         "error": "Size does not match the selected processed-window shape" if remainder else None})
    return {"path": str(root), "parent": str(root.parent), "folders": folders, "bins": bins,
            "trajectories": sum(entry["trajectories"] for entry in bins if not entry["error"])}
