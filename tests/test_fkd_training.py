"""Training entry-point tests without running an optimizer or training job."""
import argparse
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, random_split

from optcar import fkd_train


def arguments(*values):
    return fkd_train.add_training_arguments(argparse.ArgumentParser()).parse_args(values)


def dataset_factory(tmp_path):
    path = tmp_path / "dataset_config.py"
    path.write_text("from optcar.dataloader_config import DataloaderConfig\n"
                    "def dataloader_config():\n"
                    "    return DataloaderConfig(dataset_path=['main'], num_runs=-1, "
                    "test_dataset_path=['test'], test_num_runs=-1, device='cpu', no_plot=True)\n")
    return path


@pytest.fixture
def mocked_runtime(monkeypatch):
    calls = SimpleNamespace(models=[], datasets=[], trainers=[], normalization=0, loggers=[])
    monkeypatch.setattr(fkd_train, "_raise_open_file_limit", lambda: None)

    class SelectedDataset(Dataset):
        def __init__(self, dl_config, future_horizon_steps, history_horizon_steps, no_map, is_test_set):
            self.config, self.history, self.future = dl_config, history_horizon_steps, future_horizon_steps
            self.is_test_set = is_test_set
            self.files = [f"{is_test_set}.bin"]
            calls.datasets.append(self)

        def __len__(self):
            return 4 if self.is_test_set else 20

        def __getitem__(self, index):
            return {"history": torch.zeros(self.history, 13), "delta_history": torch.zeros(self.history, 13),
                    "future": torch.zeros(self.future, 13), "delta_future": torch.zeros(self.future, 13),
                    "history_actions": torch.zeros(self.history, 2), "future_actions": torch.zeros(self.future, 2)}

    class Model(torch.nn.Module):
        def __init__(self, model_config, learning_config, **kwargs):
            super().__init__()
            self.model_config, self.learning_config = model_config, learning_config
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.hparams = {}
            calls.models.append(self)

        def collect_normalization_stats(self, loader):
            calls.normalization += 1

    class Callback:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.best_model_path = ""
            self.last_model_path = ""

    class Logger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.experiment = SimpleNamespace(add_text=lambda *args: None)
            calls.loggers.append(self)

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs, self.fit_call, self.validation_call = kwargs, None, None
            calls.trainers.append(self)

        def fit(self, model, train_loader, val_loaders, ckpt_path):
            self.fit_call = (model, train_loader, val_loaders, ckpt_path)

        def validate(self, model, val_loader, verbose):
            self.validation_call = (model, val_loader, verbose)

    from optcar.models.fkd.config import dtype_to_trainer_precision
    runtime = SimpleNamespace(L=SimpleNamespace(Trainer=Trainer, __version__="test",
                                               seed_everything=lambda seed, workers: torch.manual_seed(seed)),
                              FKDDataset=SelectedDataset, ModelCheckpoint=Callback, EarlyStopping=Callback,
                              TensorBoardLogger=Logger, dtype_to_trainer_precision=dtype_to_trainer_precision,
                              models={"TransformerModel_decoderonly_CTX": Model})
    monkeypatch.setattr(fkd_train, "load_runtime", lambda: runtime)
    return calls


def test_reference_split_and_ordinary_sampler():
    dataset = list(range(37))
    actual = fkd_train.split_dataset(dataset, .9, 1 - .9, 0, seed=42)
    sizes = [int(.9 * 37), int((1 - .9) * 37)]
    expected = random_split(dataset, [*sizes, 37 - sum(sizes), 0], generator=torch.Generator().manual_seed(42))
    assert [part.indices for part in actual] == [part.indices for part in expected]
    train = fkd_train.get_dataloader(actual[0], 4, 0, shuffle=True, is_train=True)
    val = fkd_train.get_dataloader(actual[1], 4, 0)
    assert isinstance(train.sampler, RandomSampler) and not train.sampler.replacement
    assert isinstance(val.sampler, SequentialSampler)
    assert not train.pin_memory and val.pin_memory
    assert sorted(list(train.sampler)) == list(range(len(actual[0])))


def test_training_raises_open_file_limit_to_reference_budget(monkeypatch):
    import resource

    changes = []
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1_048_576))
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: changes.append((which, limits)))
    fkd_train._raise_open_file_limit()
    assert changes == [(resource.RLIMIT_NOFILE, (65_536, 1_048_576))]


def test_actual_recipe_setup_uses_reference_effects_without_training(tmp_path, mocked_runtime):
    args = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--device", "cpu", "--output", str(tmp_path / "run"))
    result = fkd_train.run_training(args)
    trainer = mocked_runtime.trainers[0]
    model, train, val_loaders, checkpoint = trainer.fit_call
    assert trainer.kwargs["max_epochs"] == 500
    assert trainer.kwargs["precision"] == "bf16-mixed"
    assert "gradient_clip_val" not in trainer.kwargs
    assert train.batch_size == 4096 and train.num_workers == 4
    assert train.prefetch_factor == 4 and train.persistent_workers and not train.pin_memory
    assert val_loaders[0].num_workers == 4 and val_loaders[1].num_workers == 1
    assert isinstance(train.sampler, RandomSampler) and not train.sampler.replacement
    assert model.dataset_size == result["training_windows"] == 18
    assert result["validation_windows"] == 1
    expected = fkd_train.split_dataset(list(range(20)), .9, 1 - .9, 0, seed=42)
    assert train.dataset.indices == sorted(expected[0].indices)
    assert val_loaders[0].dataset.indices == sorted(expected[1].indices)
    assert mocked_runtime.datasets[0].config.file_selection_seed == 42
    assert mocked_runtime.normalization == 1 and checkpoint is None
    assert model.weight.tolist() == [1.0]
    saved = torch.load(tmp_path / "run" / "initial.ckpt", weights_only=True)
    assert set(saved) == {"state_dict", "hyper_parameters", "epoch", "global_step", "pytorch-lightning_version"}
    effective = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert effective["dataloader"]["batch_size"] == 4096
    assert effective["model"]["device"] == "cpu"
    assert len(mocked_runtime.loggers) == 1


def test_explicit_wandb_logs_resolved_configs_without_external_initialization(tmp_path, mocked_runtime, monkeypatch):
    import lightning.pytorch.loggers
    wandb_calls = []
    records = []

    class WandbLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.experiment = SimpleNamespace(config=SimpleNamespace(
                update=lambda record, **options: records.append((record, options))))
            wandb_calls.append(self)

    monkeypatch.setattr(lightning.pytorch.loggers, "WandbLogger", WandbLogger)
    args = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--device", "cpu",
                     "--output", str(tmp_path / "run"), "--seed", "9", "--run-name", "wandb-test",
                     "--wandb", "--wandb-project-name", "test-project")
    fkd_train.run_training(args)
    logger = wandb_calls[0]
    assert mocked_runtime.trainers[0].kwargs["logger"] == [mocked_runtime.loggers[0], logger]
    assert logger.kwargs["project"] == "test-project"
    assert logger.kwargs["name"].endswith("/wandb-test-1")
    assert logger.kwargs["sync_tensorboard"] is True and logger.kwargs["log_model"] is False
    assert len(records) == 3 and all(options == {"allow_val_change": True} for _, options in records)
    resolved = {key: value for record, _ in records for key, value in record.items()}
    assert set(resolved) == {"DataloaderConfig", "FKD_model_config", "SupervisedLearningConfig"}
    assert resolved["DataloaderConfig"]["batch_size"] == 4096
    assert resolved["DataloaderConfig"]["file_selection_seed"] == 9
    assert resolved["FKD_model_config"]["device"] == "cpu"
    assert resolved["SupervisedLearningConfig"]["seed"] == 9
    assert resolved["SupervisedLearningConfig"]["epochs"] == 500


def test_check_validates_shapes_and_never_fits_or_writes(tmp_path, mocked_runtime):
    output = tmp_path / "must-not-exist"
    checkpoint = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor([7.0])}}, checkpoint)
    args = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--checkpoint", str(checkpoint),
                     "--device", "cpu", "--output", str(output), "--check", "--seed", "9")
    result = fkd_train.run_training(args)
    assert result["checked"] and result["sample_shapes"]["history"] == [250, 13]
    assert result["sample_shapes"]["future"] == [50, 13]
    assert result["file_selection_seed"] == result["seed"] == 9
    assert result["test_windows"] == 4
    assert mocked_runtime.models[0].weight.tolist() == [7.0]
    assert not output.exists()
    assert not mocked_runtime.trainers and not mocked_runtime.loggers
    assert mocked_runtime.normalization == 0


def test_gradient_clipping_is_enabled_only_by_learning_factory(tmp_path, mocked_runtime):
    learning = tmp_path / "learning.py"
    learning.write_text((fkd_train.CONFIG_DIR / "corl26_learning.py").read_text().replace(
        "use_gradient_clip=False", "use_gradient_clip=True"))
    args = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--learning-config", str(learning),
                     "--device", "cpu", "--output", str(tmp_path / "run"))
    fkd_train.run_training(args)
    assert mocked_runtime.trainers[0].kwargs["gradient_clip_val"] == .001


def test_fine_tune_loads_weights_but_resume_passes_complete_checkpoint(tmp_path, mocked_runtime):
    checkpoint = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor([7.0])}}, checkpoint)
    first = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--checkpoint", str(checkpoint),
                      "--device", "cpu", "--output", str(tmp_path / "run"), "--seed", "9")
    fkd_train.run_training(first)
    model, _, _, fit_checkpoint = mocked_runtime.trainers[0].fit_call
    assert model.weight.tolist() == [7.0] and fit_checkpoint is None
    resume_checkpoint = tmp_path / "run" / "last.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor([7.0])}}, resume_checkpoint)
    resumed = arguments("--resume", str(resume_checkpoint), "--device", "cpu")
    fkd_train.run_training(resumed)
    assert mocked_runtime.trainers[1].fit_call[3] == str(resume_checkpoint)
    assert mocked_runtime.datasets[-2].config.file_selection_seed == 42
    with pytest.raises(ValueError, match="uses the saved config factories"):
        fkd_train.run_training(arguments("--resume", str(resume_checkpoint), "--dataset-config", str(first.dataset_config)))


def test_incompatible_checkpoint_and_invalid_sample_fail_check(tmp_path, mocked_runtime):
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": {"other_model": torch.ones(1)}}, checkpoint)
    args = arguments("--dataset-config", str(dataset_factory(tmp_path)), "--checkpoint", str(checkpoint),
                     "--device", "cpu", "--output", str(tmp_path / "output"), "--check")
    with pytest.raises(ValueError, match="does not match"):
        fkd_train.run_training(args)
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="history has shape"):
        fkd_train._validate_sample({"history": torch.zeros(249, 13)}, SimpleNamespace(history_horizon_steps=250, future_horizon_steps=50))


def test_factory_original_imports_and_missing_config_never_fall_back(tmp_path):
    path = tmp_path / "original.py"
    path.write_text("from wheeledlab_data.dataloaders.FKD.config import DataloaderConfig\n"
                    "def dataloader_config():\n    return DataloaderConfig(dataset_path=['one'], num_runs=-1)\n")
    config = fkd_train.load_config(tmp_path, path, ["dataloader_config"])
    assert type(config).__module__ == "optcar.dataloader_config"
    with pytest.raises(FileNotFoundError, match="not found"):
        fkd_train.load_config(tmp_path, "missing.py", ["dataloader_config"], object)


def test_cli_rejects_legacy_recipe_and_train_facade_matches(capsys):
    from optcar.cli import main
    from optcar import train
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", "configs/train_generalist.yaml"])
    assert exc.value.code == 2
    assert "--dataset-config" in capsys.readouterr().err
    assert train.train is train.run_training is fkd_train.run_training
    assert arguments().train_split == .9
    assert arguments().train_num_workers == arguments().val_num_workers == 4
    assert not arguments().wandb
