"""Self-contained WheeledLab FKD training entry point.

The model, losses, scheduler, split and loader setup follow WheeledLab's
``wheeledlab_learning/FKD/train.py``. Configuration factories are kept separate
from the legacy combined YAML training recipe.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, is_dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import secrets
import shutil
import socket
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset, random_split


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LEGACY_CONFIG_MESSAGE = (
    "Combined YAML training recipes are no longer supported by optcar train. "
    "Use --dataset-config configs/corl26_data.py "
    "--model-config configs/corl26_model.py "
    "--learning-config configs/corl26_learning.py; each file defines a Python config factory."
)
_CONFIG_IMPORTS = {
    "wheeledlab_data.dataloaders.FKD.config": "optcar.dataloader_config",
    "wheeledlab_models.learned.FKD.config": "optcar.models.fkd.config",
    "wheeledlab_learning.FKD.learning_config": "optcar.models.fkd.learning_config",
}


def add_training_arguments(parser):
    """Shared options for ``optcar train`` and ``scripts/train.py``."""
    parser.add_argument("--dataset-config", "--dl-config", "--dl_config", dest="dataset_config",
                        help="Python factory defining dataloader_config()")
    parser.add_argument("--model-config", "--m_config", dest="model_config", default=None,
                        help="Python model_config() factory; defaults to configs/corl26_model.py")
    parser.add_argument("--learning-config", "--l_config", dest="learning_config", default=None,
                        help="Python learning_config() factory; defaults to configs/corl26_learning.py")
    parser.add_argument("--config", help="Legacy combined YAML recipe (unsupported)")
    parser.add_argument("--architecture", help=argparse.SUPPRESS)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint", "--ckpt_path", dest="checkpoint",
                             help="Initialize model weights for a new run; reset optimizer and scheduler")
    checkpoints.add_argument("--resume", help="Resume a Lightning checkpoint using its saved config factories")
    parser.add_argument("--output", help="Checkpoint directory for this run")
    parser.add_argument("--device", "--model_device", dest="device", help="cpu, cuda:N, or a GPU index")
    parser.add_argument("--data-device", "--data_device", dest="data_device")
    parser.add_argument("--seed", "--l_seed", dest="seed", type=int)
    parser.add_argument("--run-name", "--run_name", dest="run_name")
    parser.add_argument("--train-split", "--train_split", dest="train_split", type=float, default=0.9)
    parser.add_argument("--train-num-workers", "--train_num_workers", dest="train_num_workers", type=int, default=4)
    parser.add_argument("--val-num-workers", "--val_num_workers", dest="val_num_workers", type=int, default=4)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-traj", "--limit_traj", dest="limit_traj", type=int, default=-1)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--train-as-val", "--train_as_val", dest="train_as_val", action="store_true")
    parser.add_argument("--no-val", "--no_val", dest="no_val", action="store_true")
    parser.add_argument("--no-test", "--no_test", dest="no_test", action="store_true")
    parser.add_argument("--dataloader-warmup", action="store_true")
    parser.add_argument("--fastdev", action="store_true", help="Lightning fast_dev_run")
    parser.add_argument("--check", "--dry-run", dest="check", action="store_true",
                        help="Validate configs, selected data and checkpoint without training or writing output")
    parser.add_argument("--wandb", action="store_true", help="Enable external WandB logging (off by default)")
    parser.add_argument("--no-wandb", "--no_wandb", dest="wandb", action="store_false", default=False)
    parser.add_argument("--wandb-project-name", default="wheeledlab")
    return parser


def resolve_config_path(filename, base_dir=CONFIG_DIR):
    if filename is None:
        raise ValueError("Provide --dataset-config with a dataloader_config() Python factory.")
    entered = Path(filename).expanduser()
    candidates = [entered, Path(base_dir) / entered]
    if entered.suffix != ".py":
        candidates.extend([Path(str(entered) + ".py"), Path(base_dir) / (str(entered) + ".py")])
    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix != ".py":
                raise ValueError(LEGACY_CONFIG_MESSAGE)
            return candidate.resolve()
    raise FileNotFoundError(f"Config factory not found: {filename}")


def load_config(base_dir, filename, candidates, fallback_class=None):
    """Load the reference factory, translating its original package imports.

    Saved WheeledLab factories can be reused without installing WheeledLab.
    Unlike the original entry point, missing factories fail instead of silently
    selecting unrelated default configurations.
    """
    path = resolve_config_path(filename, base_dir)
    spec = importlib.util.spec_from_file_location(f"optcar_fkd_config_{secrets.token_hex(4)}", path)
    module = importlib.util.module_from_spec(spec)
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _CONFIG_IMPORTS:
            node.module = _CONFIG_IMPORTS[node.module]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    for name in candidates:
        if hasattr(module, name):
            config = getattr(module, name)()
            print(f"Loaded config from {path} using function '{name}'")
            return config
    raise ValueError(f"Config file {path} must define one of: {', '.join(candidates)}")


def parse_device(device_label, device_str):
    if str(device_str).lower() == "cpu":
        return "cpu"
    if str(device_str).startswith("cuda"):
        return int(str(device_str).split(":", 1)[1]) if ":" in str(device_str) else 0
    try:
        return int(device_str)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{device_label} must be 'cpu', 'cuda:N', or an integer") from exc


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """WheeledLab's seeded random split over selected trajectory windows."""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    residue = max(0, dataset_size - train_size - val_size - test_size)
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size, residue], generator=generator)


def get_dataloader(dataset, batch_size, num_workers, shuffle=False, learning_config=None, is_train=False):
    """WheeledLab loader settings: ordinary shuffle, no terrain sampler."""
    device = getattr(dataset, "device", None)
    dataset_on_cpu = (device is None or isinstance(device, str) and device.lower() == "cpu"
                      or isinstance(device, torch.device) and device.type == "cpu")
    if not isinstance(device, (str, torch.device, int)):
        dataset_on_cpu = True
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                  num_workers=num_workers, pin_memory=(not is_train) and dataset_on_cpu)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
        if not dataset_on_cpu:
            kwargs["multiprocessing_context"] = mp.get_context("spawn")
    if not dataset_on_cpu:
        kwargs["pin_memory"] = False
    return DataLoader(**kwargs)


def load_runtime():
    """Import training dependencies only when the training command is used."""
    try:
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
        from optcar.fkd_dataloader import FKDDataset
        from optcar.models.fkd.config import dtype_to_trainer_precision
        from optcar.models.fkd.transformer import TransformerModel_decoderonly, TransformerModel_decoderonly_CTX
    except ImportError as exc:
        raise ImportError("FKD training dependencies are missing. Install the packages in requirements.txt before running training.") from exc
    return SimpleNamespace(L=L, EarlyStopping=EarlyStopping, ModelCheckpoint=ModelCheckpoint,
                           TensorBoardLogger=TensorBoardLogger, FKDDataset=FKDDataset,
                           dtype_to_trainer_precision=dtype_to_trainer_precision,
                           models={"TransformerModel_decoderonly": TransformerModel_decoderonly,
                                   "TransformerModel_decoderonly_CTX": TransformerModel_decoderonly_CTX})


def _config_dict(config):
    return asdict(config) if is_dataclass(config) else vars(config)


def save_initial_checkpoint(model, path, lightning_version):
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": getattr(model, "hparams", {}),
                "epoch": -1, "global_step": 0, "pytorch-lightning_version": lightning_version}, path)


def load_checkpoint_weights(path):
    """Read original Lightning state dictionaries without general pickle execution."""
    from lightning.fabric.utilities.data import AttributeDict
    with torch.serialization.safe_globals([AttributeDict]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("state_dict", payload)


def _checkpoint_and_configs(args):
    if args.config or args.architecture:
        raise ValueError(LEGACY_CONFIG_MESSAGE)
    checkpoint = Path(args.resume or args.checkpoint).expanduser().resolve() if (args.resume or args.checkpoint) else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if args.resume:
        paths = [checkpoint.parent / name for name in ("dl_config.py", "model_config.py", "learning_config.py")]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("Resuming requires dl_config.py, model_config.py and learning_config.py beside the checkpoint.")
        if args.output and Path(args.output).expanduser().resolve() != checkpoint.parent:
            raise ValueError("--resume writes to its checkpoint directory; omit --output or use the same directory.")
        if any((args.dataset_config, args.model_config, args.learning_config)):
            raise ValueError("--resume uses the saved config factories; use --checkpoint for a new dataset or configuration.")
    else:
        inherited_model = checkpoint.parent / "model_config.py" if checkpoint is not None else None
        model = args.model_config or (inherited_model if inherited_model is not None and inherited_model.is_file() else "corl26_model.py")
        paths = [resolve_config_path(args.dataset_config), resolve_config_path(model),
                 resolve_config_path(args.learning_config or "corl26_learning.py")]
    return checkpoint, paths


def _prepare_loaders(runtime, args, data, learning, model_config, model):
    dataset_kwargs = dict(dl_config=data, future_horizon_steps=model_config.future_horizon_steps,
                          history_horizon_steps=model_config.history_horizon_steps, no_map=model_config.no_map)
    full_dataset = runtime.FKDDataset(**dataset_kwargs, is_test_set=False)
    main_files = getattr(full_dataset, "files", [])
    if 0 < args.limit_traj < len(full_dataset):
        generator = torch.Generator().manual_seed(learning.seed)
        full_dataset = Subset(full_dataset, torch.randperm(len(full_dataset), generator=generator)[:args.limit_traj].tolist())
    train_dataset, val_dataset, test_dataset, _ = split_dataset(full_dataset, args.train_split, 1 - args.train_split, 0.0, learning.seed)
    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.indices = sorted(dataset.indices)
    model.dataset_size = len(train_dataset)
    if not model.dataset_size:
        raise ValueError("The selected dataset and --train-split produce an empty training set.")
    if args.train_as_val:
        val_dataset = train_dataset
    if not args.no_val and not len(val_dataset):
        raise ValueError("The selected dataset and --train-split produce an empty validation set.")
    train_loader = get_dataloader(train_dataset, data.batch_size, max(0, args.train_num_workers),
                                  shuffle=not args.no_shuffle, learning_config=learning, is_train=True)
    val_loader = get_dataloader(val_dataset, data.batch_size, max(0, args.val_num_workers), learning_config=learning)
    val_loaders = [] if args.no_val else [val_loader]
    test_paths = getattr(data, "test_dataset_path", None)
    if not args.no_test and test_paths:
        test_dataset = runtime.FKDDataset(**dataset_kwargs, is_test_set=True)
        overlap = {Path(file).resolve() for file in main_files} & {Path(file).resolve() for file in getattr(test_dataset, "files", [])}
        if overlap:
            raise ValueError(f"Main and test data select the same binary files: {sorted(overlap)}. Select disjoint recorded splits.")
        val_loaders.append(get_dataloader(test_dataset, data.batch_size, 1, learning_config=learning))
    return train_loader, val_loader, val_loaders, train_dataset, val_dataset


def _validate_sample(sample, model_config):
    history, future = model_config.history_horizon_steps, model_config.future_horizon_steps
    expected = {"history": (history, 13), "delta_history": (history, 13),
                "future": (future, 13), "delta_future": (future, 13),
                "history_actions": (history, 2), "future_actions": (future, 2)}
    for name, shape in expected.items():
        value = sample.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(f"Dataset sample {name} has shape {actual}; the FKD model requires {shape}.")


def run_training(args):
    """Execute the reference Lightning training setup with local OptCar modules."""
    checkpoint, config_paths = _checkpoint_and_configs(args)
    if not 0 < args.train_split <= 1:
        raise ValueError("--train-split must lie in (0, 1].")
    runtime = load_runtime()
    torch.set_float32_matmul_precision("medium")
    learning = load_config(CONFIG_DIR, config_paths[2], ["learning_config", "get_learning_config", "config"])
    if args.seed is not None:
        learning.seed = args.seed
    if args.epochs is not None:
        learning.epochs = args.epochs
    data = load_config(CONFIG_DIR, config_paths[0], ["dataloader_config", "get_dataloader_config", "config"])
    if getattr(data, "file_selection_seed", None) is None:
        data.file_selection_seed = learning.seed
    if args.batch_size is not None:
        data.batch_size = args.batch_size
    if data.batch_size is None:
        data.batch_size = getattr(learning, "batch_size", 32)
    model_config = load_config(CONFIG_DIR, config_paths[1], ["model_config", "get_model_config", "config"])
    if args.device is not None:
        model_config.device = parse_device("model device", args.device)
    if args.data_device is not None:
        data.device = parse_device("data device", args.data_device)
    if min(learning.epochs, data.batch_size) < 1:
        raise ValueError("Epochs and batch size must be positive.")
    device = model_config.device
    requested = device[0] if isinstance(device, (list, tuple)) and device else device
    if isinstance(requested, int):
        if not torch.cuda.is_available() or not 0 <= requested < torch.cuda.device_count():
            raise ValueError(f"Requested cuda:{requested} is unavailable; select --device cpu or an available GPU.")
        torch.cuda.set_device(requested)
    runtime.L.seed_everything(learning.seed, workers=True)
    if learning.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        model_class = runtime.models[model_config.model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported FKD model_type: {model_config.model_type}") from exc
    model = model_class(model_config, learning,
                        plot_val_epoch_freq=data.plot_val_epoch_freq, plot_val_batch_freq=data.plot_val_batch_freq,
                        plot_test_epoch_freq=data.plot_test_epoch_freq, plot_test_batch_freq=data.plot_test_batch_freq,
                        no_plot=data.no_plot)
    if checkpoint is not None and (not args.resume or args.check):
        try:
            model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Checkpoint does not match the requested FKD model configuration: {exc}") from exc
        if not args.resume:
            print(f"Initialized model weights from {checkpoint}; optimizer and scheduler reset for a new run.")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    label = f"{round(trainable / 1_000_000)}M" if trainable >= 1_000_000 else f"{round(trainable / 1_000)}K" if trainable >= 1_000 else str(trainable)
    run_name = f"{datetime.now():%Y-%m-%d}/{socket.gethostname().split('.')[0]}/{args.run_name or secrets.token_hex(4)}-{label}"
    runs_root = Path(getattr(learning, "checkpoint_folder", None) or Path.cwd() / "training_runs").expanduser().resolve()
    output = checkpoint.parent if args.resume else Path(args.output).expanduser().resolve() if args.output else runs_root / "checkpoints" / run_name
    train_loader, val_loader, val_loaders, train_dataset, val_dataset = _prepare_loaders(runtime, args, data, learning, model_config, model)
    if args.check:
        sample = train_dataset[0]
        _validate_sample(sample, model_config)
        return {"checked": True, "output": str(output), "checkpoint": str(checkpoint) if checkpoint else None,
                "model_type": model_config.model_type, "trainable_parameters": trainable,
                "training_windows": len(train_dataset), "validation_windows": len(val_dataset),
                "test_windows": sum(len(loader.dataset) for loader in val_loaders if loader is not val_loader),
                "batch_size": data.batch_size, "seed": learning.seed, "file_selection_seed": data.file_selection_seed,
                "sample_shapes": {key: list(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)},
                "validation_monitor_caveat": "Reference val/learn/loss_epoch stays zero; preserved for compatibility."}
    if not args.resume and output.exists() and any(output.glob("*.ckpt")):
        raise ValueError(f"Output already contains checkpoints: {output}. Choose a fresh --output or use --resume.")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for source, name in zip(config_paths, ("dl_config.py", "model_config.py", "learning_config.py")):
            if source.resolve() != (output / name).resolve():
                shutil.copy2(source, output / name)
        save_initial_checkpoint(model, output / "initial.ckpt", getattr(runtime.L, "__version__", "unknown"))
    with (output / "run_config.json").open("w") as stream:
        json.dump({"arguments": vars(args), "dataloader": _config_dict(data),
                   "model": _config_dict(model_config), "learning": _config_dict(learning)},
                  stream, indent=2, default=str, allow_nan=False)
    monitor = getattr(learning, "early_stop_monitor", "val/learn/loss_epoch")
    if monitor == "val/learn/loss":
        monitor = "val/learn/loss_epoch"
        learning.early_stop_monitor = monitor
    best = runtime.ModelCheckpoint(dirpath=str(output), filename=f"best-{{epoch:02d}}-{{{monitor.replace('/', '_')}:.4f}}",
                                   monitor=monitor, mode="min", save_top_k=1, save_weights_only=False, save_last=False)
    regular = runtime.ModelCheckpoint(dirpath=str(output), filename="epoch-{epoch:02d}", save_top_k=-1,
                                      every_n_epochs=max(1, int(learning.save_checkpoint_every_n_epoch)),
                                      save_weights_only=False, save_last=True)
    callbacks, early = [best, regular], None
    if learning.early_stopping:
        early = runtime.EarlyStopping(monitor=monitor, min_delta=learning.early_stop_min_delta,
                                      patience=learning.early_stop_patience, verbose=False,
                                      mode=learning.early_stop_mode, strict=True, check_finite=True)
        best.mode = learning.early_stop_mode
        callbacks.append(early)
    # Only local logging is enabled by default. WandB requires an explicit flag.
    log_root = output / "lightning_logs" if args.output or args.resume else runs_root / "lightning_logs"
    tb_logger = runtime.TensorBoardLogger(save_dir=str(log_root), name="run" if args.output or args.resume else run_name)
    loggers = [tb_logger]
    for config, name in ((data, "DataloaderConfig"), (model_config, "FKD_model_config"), (learning, "SupervisedLearningConfig")):
        tb_logger.experiment.add_text(name, json.dumps(_config_dict(config), indent=2, default=str), 0)
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        wb = WandbLogger(project=args.wandb_project_name, name=run_name, save_dir=str(log_root),
                         log_model=False, sync_tensorboard=True)
        loggers.append(wb)
        for config, name in ((data, "DataloaderConfig"), (model_config, "FKD_model_config"), (learning, "SupervisedLearningConfig")):
            wb.experiment.config.update({name: _config_dict(config)}, allow_val_change=True)
    if args.dataloader_warmup:
        for loader in [train_loader, *val_loaders]:
            for _ in loader:
                pass
    if getattr(model_config, "normalization", None):
        model.collect_normalization_stats(train_loader)
    cpu = requested == "cpu" or isinstance(requested, torch.device) and requested.type == "cpu"
    trainer_kwargs = dict(accelerator="cpu" if cpu else "gpu", max_epochs=learning.epochs,
                          devices=1 if cpu else [device] if isinstance(device, int) else device,
                          enable_checkpointing=True, callbacks=callbacks, fast_dev_run=args.fastdev,
                          logger=loggers, precision=runtime.dtype_to_trainer_precision(getattr(model_config, "dtype", None)),
                          check_val_every_n_epoch=data.check_val_every_n_epoch)
    if learning.use_gradient_clip and learning.gradient_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = learning.gradient_clip_val
    trainer = runtime.L.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loaders, ckpt_path=str(checkpoint) if args.resume else None)
    if early and best.best_model_path:
        model = model.__class__.load_from_checkpoint(best.best_model_path, model_config=model_config, learning_config=learning)
    # Match the reference's final evaluation of the internal validation split.
    trainer.validate(model, val_loader, verbose=True)
    return {"output": str(output), "best_checkpoint": best.best_model_path,
            "last_checkpoint": regular.last_model_path, "training_windows": len(train_dataset),
            "validation_windows": len(val_dataset)}


def main(argv=None):
    parser = add_training_arguments(argparse.ArgumentParser(description="OptCar WheeledLab FKD training"))
    args = parser.parse_args(argv)
    try:
        result = run_training(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"optcar train: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
