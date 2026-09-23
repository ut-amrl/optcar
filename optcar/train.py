"""Generalist pretraining and real/synthetic specialist fine-tuning."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from optcar.dataset import SCHEMA, DataConfig, FKDDataset, TerrainSampler
from optcar.models.fkd_transformer import FKDTransformer, ModelConfig, load_checkpoint, save_checkpoint
from optcar.evaluate import evaluate_model, prediction_loss, to_device
from optcar.utils.file_io import write_json


@dataclass
class TrainConfig:
    stage: str = "generalist"
    datasets: list = field(default_factory=lambda: ["datasets/generalist_dataset"])
    output: str = "checkpoints/film_generalist"
    checkpoint: str | None = None
    resume: str | None = None
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 10
    gradient_clip: float = 1.0
    workers: int = 0
    device: str = "cpu"
    seed: int = 42
    real_fraction: float = 0.5
    samples_per_epoch: int | None = None
    normalization_samples: int = 10000
    position_weight: float = 1.0
    quaternion_weight: float = 0.3

    def __post_init__(self):
        if self.stage not in ("generalist", "specialist"):
            raise ValueError("stage must be generalist or specialist")
        if min(self.epochs, self.batch_size, self.normalization_samples) < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid training budget or learning rate")
        if self.workers < 0 or self.warmup_epochs < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid worker count, warmup, or gradient limit")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        if self.stage == "specialist" and not (self.checkpoint or self.resume):
            raise ValueError("Specialist training requires a generalist --checkpoint")


def normalization_statistics(dataset, limit, batch_size=256):
    count = min(limit, len(dataset))
    # Evenly spaced samples across the training set; never read held-out data.
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
    totals = {}
    for batch in loader:
        for name in ("history", "current", "commands"):
            values = batch[name].reshape(-1, batch[name].shape[-1]).double()
            if name not in totals:
                totals[name] = [0, torch.zeros(values.shape[-1], dtype=torch.float64), torch.zeros(values.shape[-1], dtype=torch.float64)]
            stats = totals[name]
            stats[0] += len(values)
            stats[1] += values.sum(0)
            stats[2] += values.square().sum(0)
    result = {}
    for name, (n, sums, squares) in totals.items():
        mean = sums / n
        std = (squares / n - mean.square()).clamp_min(0).sqrt()
        # Constant channels stay unscaled, including planar zero coordinates.
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        result[name] = {"mean": mean.float().tolist(), "std": std.float().tolist(), "count": n}
    return result


def train(config, data=None, model_config=None, progress=None):
    data, model_config = data or DataConfig(), model_config or ModelConfig()
    if (model_config.history_steps, model_config.future_steps) != (data.history_steps, data.future_steps):
        raise ValueError("Model and dataset horizons must match")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    output = Path(config.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if not config.resume and (output / f"{config.stage}_last.ckpt").exists():
        raise ValueError("Output already contains a training run; choose another output or use --resume")
    # Fail before starting if either split is missing, including empty generalist data.
    dataset = FKDDataset(config.datasets, data, "train")
    validation = FKDDataset(config.datasets, data, "val", sources=["real"] if config.stage == "specialist" else None)
    if config.stage == "specialist":
        sources = {shard[2] for shard in dataset.shards}
        if not {"real", "sim"}.issubset(sources):
            raise ValueError("Specialist training requires both real and simulated training windows")
        if "real" not in {shard[2] for shard in validation.shards}:
            raise ValueError("Specialist validation requires held-out real bags")
    previous = None
    source_checkpoint = config.resume or config.checkpoint
    if source_checkpoint:
        model, saved_data, previous = load_checkpoint(source_checkpoint, config.device)
        if asdict(model.config) != asdict(model_config):
            raise ValueError("Requested model configuration does not match checkpoint architecture")
        if (saved_data.history_steps, saved_data.future_steps, saved_data.dt) != (data.history_steps, data.future_steps, data.dt):
            raise ValueError("Checkpoint timestep or horizons differ from the dataset")
        if config.stage == "specialist" and not config.resume and previous["stage"] != "generalist":
            raise ValueError("Initialize specialist training from a generalist checkpoint")
        if config.resume and previous["stage"] != config.stage:
            raise ValueError("Resume stage differs; use checkpoint initialization for fine-tuning")
        statistics = previous["normalization"]
    else:
        model = FKDTransformer(model_config).to(config.device)
        statistics = normalization_statistics(dataset, config.normalization_samples, config.batch_size)
        model.set_normalization(statistics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sampler = TerrainSampler(dataset, config.samples_per_epoch or len(dataset), config.seed, config.real_fraction)
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.workers,
                        pin_memory=config.device.startswith("cuda"), persistent_workers=config.workers > 0)
    val_loader = DataLoader(validation, batch_size=config.batch_size, num_workers=config.workers)
    total_steps = config.epochs * len(loader)
    warmup_steps = min(config.warmup_epochs * len(loader), max(0, total_steps - 1))
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        fraction = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    epoch_start, best = 0, float("inf")
    if config.resume:
        old = previous["training_config"]
        for key in ("epochs", "batch_size", "samples_per_epoch", "learning_rate", "warmup_epochs", "datasets", "seed", "real_fraction", "weight_decay", "position_weight", "quaternion_weight"):
            if old[key] != asdict(config)[key]:
                raise ValueError(f"Resume requires unchanged {key}; initialize a new run to change it")
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
        epoch_start, best = previous["epoch"] + 1, previous["best_loss"]
        torch.set_rng_state(previous["torch_rng_state"])
        if config.device.startswith("cuda") and previous.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(previous["cuda_rng_state"])
    write_json(output / "run_config.json", {"training": asdict(config), "data": asdict(data), "model": asdict(model_config)})
    write_json(output / "normalization.json", statistics)
    if config.resume and not (output / f"{config.stage}_best.ckpt").exists():
        prior_best = Path(config.resume).expanduser().parent / f"{config.stage}_best.ckpt"
        if prior_best.exists() and prior_best.resolve() != (output / prior_best.name).resolve():
            import shutil
            shutil.copy2(prior_best, output / prior_best.name)
    for epoch in range(epoch_start, config.epochs):
        sampler.epoch = epoch
        model.train()
        loss_sum, count = 0., 0
        for raw in loader:
            batch = to_device(raw, config.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["history"], batch["current"], batch["commands"])
            loss = prediction_loss(prediction, batch, config.position_weight, config.quaternion_weight)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            size = len(batch["current"])
            loss_sum += float(loss.detach()) * size
            count += size
        metrics = evaluate_model(model, val_loader, config.device, config.position_weight, config.quaternion_weight)
        if not math.isfinite(metrics["loss"]):
            raise ValueError("Validation loss became nonfinite")
        metrics.update(epoch=epoch, training_loss=loss_sum / count, learning_rate=optimizer.param_groups[0]["lr"])
        improved = metrics["loss"] < best
        best = min(best, metrics["loss"])
        checkpoint = {"format_version": 1, "schema": SCHEMA, "stage": config.stage,
            "model_config": asdict(model_config), "data_config": asdict(data), "training_config": asdict(config),
            "model_state": model.state_dict(), "normalization": statistics, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch, "best_loss": best,
            "metrics": metrics, "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if config.device.startswith("cuda") else [],
            "parent_checkpoint": config.checkpoint}
        save_checkpoint(output / f"{config.stage}_last.ckpt", checkpoint)
        if improved:
            save_checkpoint(output / f"{config.stage}_best.ckpt", checkpoint)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if progress:
            progress(metrics)
    return {"output": str(output), "best_loss": best, "epochs": config.epochs}
