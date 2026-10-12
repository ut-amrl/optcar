"""FKD architecture and numerical parity against independently loaded source."""
import ast
import copy
import logging
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import typing

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

L = pytest.importorskip("lightning")

from optcar.models.fkd import backbone
from optcar.models.fkd.config import FKD_model_config, resolve_dtype
from optcar.models.fkd.learning_config import SupervisedLearningConfig
from optcar.models.fkd.transformer import TransformerModel_decoderonly_CTX


REPO = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("WHEELEDLAB_SOURCE", REPO.parent / "WheeledLab/source"))
MODEL_SOURCE = SOURCE / "wheeledlab_models/wheeledlab_models/learned/FKD"


def configs():
    model = runpy.run_path(str(REPO / "configs/corl26_model.py"))["model_config"]()
    learning = runpy.run_path(str(REPO / "configs/corl26_learning.py"))["learning_config"]()
    model.device = "cpu"
    return model, learning


def _execute_definitions(path, namespace, names):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        assigned = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
        if getattr(node, "name", None) in names or any(name in names for name in assigned):
            node = copy.deepcopy(node)
            # Reference math executes eagerly, avoiding compilation in parity tests.
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)


@pytest.fixture
def reference_class():
    if not MODEL_SOURCE.is_dir():
        pytest.skip("Local WheeledLab source is unavailable")
    ns = dict(vars(typing))
    ns.update(__name__="fkd_reference", torch=torch, nn=nn, optim=torch.optim, F=F, np=np, L=L, logging=logging,
              SimpleNamespace=SimpleNamespace, FKD_model_config=FKD_model_config,
              SupervisedLearningConfig=SupervisedLearningConfig, resolve_dtype=resolve_dtype)
    _execute_definitions(SOURCE / "wheeledlab_utils/wheeledlab_utils/math.py", ns,
                         {"cross_product_3d", "qrotate_torch", "qmultiply_torch"})
    _execute_definitions(SOURCE / "wheeledlab_utils/wheeledlab_utils/filter.py", ns,
                         {"quaternion_continuity_align", "quaternion_continuity_align_torchscript"})
    _execute_definitions(SOURCE / "wheeledlab_utils/wheeledlab_utils/transform_utils.py", ns,
                         {"reconstruct_absolute_from_deltas"})
    _execute_definitions(MODEL_SOURCE / "transformer_helper.py", ns,
                         {"LearnedPositionalEncoding", "MultiLinear", "MapEncoder_ViT"})
    _execute_definitions(MODEL_SOURCE / "backbone.py", ns,
                         {"_NORMALIZATION_STRATEGIES", "_STATE_COMPONENTS_13", "_STATE_COMPONENTS_7",
                          "_ACTION_COMPONENTS_2", "_default_learning_config", "_coerce_learning_config",
                          "_component_names_for_field", "NormalizationGroup", "FieldNormalizer",
                          "quat_ambiguity_resolved_loss", "quat_geodesic_loss", "TrajectoryLosses",
                          "FKDLightningModule"})
    _execute_definitions(MODEL_SOURCE / "transformer.py", ns,
                         {"TransformerModel_decoderonly", "FiLMLayer", "TransformerModel_decoderonly_CTX"})
    return ns["TransformerModel_decoderonly_CTX"]


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


def test_active_source_definitions_are_unchanged():
    if not MODEL_SOURCE.is_dir():
        pytest.skip("Local WheeledLab source is unavailable")

    def definitions(path, names=None):
        nodes = [node for node in ast.parse(path.read_text()).body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
        if names is not None:
            nodes = [node for node in nodes if getattr(node, "name", None) in names]
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)

    target = REPO / "optcar/models/fkd"
    for name in ("config.py", "backbone.py", "transformer_helper.py"):
        assert definitions(target / name) == definitions(MODEL_SOURCE / name)
    active = {"TransformerModel_decoderonly", "FiLMLayer", "TransformerModel_decoderonly_CTX"}
    assert definitions(target / "transformer.py", active) == definitions(MODEL_SOURCE / "transformer.py", active)
    learning_source = SOURCE / "wheeledlab_learning/wheeledlab_learning/FKD/learning_config.py"
    assert definitions(target / "learning_config.py") == definitions(learning_source)


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


def _models(reference_class):
    cfg, learning = configs()
    actual = TransformerModel_decoderonly_CTX(cfg, learning, dataset_size=40_960, no_plot=True)
    expected = reference_class(copy.deepcopy(cfg), copy.deepcopy(learning), dataset_size=40_960, no_plot=True)
    expected.load_state_dict(actual.state_dict(), strict=True)
    return actual, expected


def test_forward_normalization_losses_and_gradients_match(reference_class, monkeypatch):
    actual, expected = _models(reference_class)
    # The production transform retains its source compile decorator. This CPU
    # parity test compares its original eager function to an independent reference.
    reconstruct = backbone.reconstruct_absolute_from_deltas
    monkeypatch.setattr(backbone, "reconstruct_absolute_from_deltas", getattr(reconstruct, "__wrapped__", reconstruct))
    actual.eval()
    expected.eval()
    actual_batch, expected_batch = copy.deepcopy(_batch()), copy.deepcopy(_batch())
    actual_processed = actual.pre_process_data(actual_batch)
    expected_processed = expected.pre_process_data(expected_batch)
    for key in actual_processed:
        if isinstance(actual_processed[key], torch.Tensor):
            torch.testing.assert_close(actual_processed[key], expected_processed[key], rtol=0, atol=0)
    actual_output, expected_output = actual(actual_processed), expected(expected_processed)
    for key in actual_output:
        if actual_output[key] is not None:
            torch.testing.assert_close(actual_output[key], expected_output[key], rtol=0, atol=0)
    post_args = (actual_processed["vel_wrt_world"], actual_processed["angvel_wrt_world"])
    actual_post = actual.post_process_data_learning(actual_output["future_traj"], *post_args)
    expected_post = expected.post_process_data_learning(expected_output["future_traj"], *post_args)
    for key in actual_post:
        torch.testing.assert_close(actual_post[key], expected_post[key], atol=1e-6, rtol=1e-6)

    actual.train()
    expected.train()
    torch.manual_seed(84)
    actual_loss, actual_metrics, *_ = actual._step(copy.deepcopy(_batch()), "train")
    torch.manual_seed(84)
    expected_loss, expected_metrics, *_ = expected._step(copy.deepcopy(_batch()), "train")
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-6)
    assert actual_loss.requires_grad and actual_loss.item() > 0
    assert actual_metrics.keys() == expected_metrics.keys()
    actual_loss.backward()
    expected_loss.backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == reference_name
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=2e-6, rtol=2e-6)

    actual.eval()
    expected.eval()
    actual_val, actual_metrics, *_ = actual._step(copy.deepcopy(_batch()), "val")
    expected_val, expected_metrics, *_ = expected._step(copy.deepcopy(_batch()), "val")
    assert actual_val.item() == expected_val.item() == 0
    assert actual_metrics["val/learn/loss_epoch"].item() == 0
    for key in actual_metrics:
        torch.testing.assert_close(actual_metrics[key], expected_metrics[key], atol=1e-6, rtol=1e-6)


def test_optimizer_schedule_matches_source(reference_class):
    actual, expected = _models(reference_class)
    actual_setup, expected_setup = actual.configure_optimizers(), expected.configure_optimizers()
    actual_optimizer, expected_optimizer = actual_setup["optimizer"], expected_setup["optimizer"]
    actual_scheduler = actual_setup["lr_scheduler"]["scheduler"]
    expected_scheduler = expected_setup["lr_scheduler"]["scheduler"]
    assert actual_setup["lr_scheduler"]["interval"] == "step"
    assert actual_optimizer.param_groups[0]["weight_decay"] == .01
    # Cross the 1% warmup boundary of the 5,000-step source schedule.
    for _ in range(53):
        assert actual_optimizer.param_groups[0]["lr"] == expected_optimizer.param_groups[0]["lr"]
        actual_optimizer.step()
        expected_optimizer.step()
        actual_scheduler.step()
        expected_scheduler.step()
