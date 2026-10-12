"""Regression checks for the fixed FKD architecture and numerical behavior."""
import copy
from pathlib import Path
import runpy
import numpy as np
import pytest
import torch
import torch.nn.functional as F
pytest.importorskip("lightning")
from optcar.models.fkd import backbone
from optcar.models.fkd.transformer import TransformerModel_decoderonly_CTX
REPO = Path(__file__).resolve().parents[1]

def configs():
    model = runpy.run_path(str(REPO / "configs/corl26_model.py"))["model_config"]()
    learning = runpy.run_path(str(REPO / "configs/corl26_learning.py"))["learning_config"]()
    model.device = "cpu"
    return model, learning

def test_exact_321k_architecture_and_state_dict():
    cfg, learning = configs()
    model = TransformerModel_decoderonly_CTX(cfg, learning, no_plot=True)
    assert sum(p.numel() for p in model.parameters()) == 321_219
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 321_219
    state = model.state_dict()
    assert len(state) == len(dict(model.named_parameters())) == 95
    assert sum(value.numel() for value in state.values()) == 321_219
    assert state["combined_encoder_pos_enc.pos_enc"].shape == (501, 64)
    assert state["future_action_pos_enc.pos_enc"].shape == (50, 64)
    assert state["curr_state_encoder.network.0.weight"].shape == (30, 10)
    assert state["fc_out.network.4.weight"].shape == (7, 30)
    assert len(model.film_layers) == len(model.film_xattn_layers) == 2
    assert all(not layer.norm_first for layer in [*model.decoder.layers, *model.history_encoder.layers])
    assert model.combined_encoder_pos_enc.dropout.p == model.future_action_pos_enc.dropout.p == .1
    assert all(layer.dropout.p == .1 for layer in model.decoder.layers)
    assert all(layer.dropout.p == 0 for layer in model.history_encoder.layers)
    assert cfg.dtype == "bf16-mixed"
    assert model._model_dtype == torch.bfloat16
    assert {p.dtype for p in model.parameters()} == {torch.float32}

def _batch():
    generator = torch.Generator().manual_seed(83)

    def states(length):
        value = torch.randn((2, length, 13), generator=generator) * .1
        value[..., 3] += 1
        value[..., 3:7] = F.normalize(value[..., 3:7], dim=-1)
        return value

    history, future = states(250), states(50)
    delta_history, delta_future = states(250), states(50)
    delta_future[:, 0, :3] = 0
    delta_future[:, 0, 3:7] = torch.tensor([1., 0., 0., 0.])
    return {"history": history, "future": future, "delta_history": delta_history,
            "delta_future": delta_future, "history_actions": torch.randn((2, 250, 2), generator=generator),
            "future_actions": torch.randn((2, 50, 2), generator=generator), "hmap": torch.zeros((2, 1, 1)),
            "quat_wrt_world": history[:, -1, 3:7].clone(),
            "vel_wrt_world": history[:, -1, 7:10].clone(),
            "angvel_wrt_world": history[:, -1, 10:13].clone()}


def model():
    torch.manual_seed(81)
    cfg, learning = configs()
    return TransformerModel_decoderonly_CTX(cfg, learning, dataset_size=40_960, no_plot=True)


def test_forward_normalization_losses_and_gradients(oracle, monkeypatch):
    actual = model()
    reconstruct = backbone.reconstruct_absolute_from_deltas
    monkeypatch.setattr(backbone, "reconstruct_absolute_from_deltas", getattr(reconstruct, "__wrapped__", reconstruct))
    def check(prefix, values, tolerance=1e-6):
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, torch.from_numpy(oracle[prefix + key]), atol=tolerance, rtol=tolerance)
    actual.eval()
    processed = actual.pre_process_data(copy.deepcopy(_batch()))
    check("model.preprocessed.", processed, 0)
    output = actual(processed)
    check("model.output.", output)
    check("model.postprocessed.", actual.post_process_data_learning(
        output["future_traj"], processed["vel_wrt_world"], processed["angvel_wrt_world"]))
    actual.train()
    torch.manual_seed(84)
    loss, metrics, *_ = actual._step(copy.deepcopy(_batch()), "train")
    torch.testing.assert_close(loss, torch.from_numpy(oracle["model.train_loss"]), atol=1e-6, rtol=1e-6)
    assert loss.requires_grad and loss.item() > 0
    check("model.train_metric.", metrics)
    loss.backward()
    check("model.gradient.", {name: p.grad for name, p in actual.named_parameters()}, 2e-6)
    assert all(p.grad is not None for p in actual.parameters())
    actual.eval()
    val_loss, metrics, *_ = actual._step(copy.deepcopy(_batch()), "val")
    assert val_loss.item() == metrics["val/learn/loss_epoch"].item() == 0
    check("model.val_metric.", metrics)


def test_optimizer_schedule(oracle):
    setup = model().configure_optimizers()
    optimizer, scheduler = setup["optimizer"], setup["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert setup["lr_scheduler"]["interval"] == "step"
    assert optimizer.param_groups[0]["weight_decay"] == .01
    rates = []
    for _ in range(53):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    np.testing.assert_array_equal(rates, oracle["model.optimizer_lr_schedule"])
