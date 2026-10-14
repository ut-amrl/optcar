"""Export an OptCar Lightning checkpoint to ONNX and TensorRT.

Run with the target GPU's TensorRT environment:
    python3 optcar/export_trt.py --checkpoint /path/to/last.ckpt --max-batch 601

The wrapper and dynamic-batch build follow the FKD export pipeline. The engine
contains the network forward pass; the controller applies checkpoint input and
output normalization. ONNX and engine files are saved beside the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from optcar.models.fkd.transformer import TransformerModel_decoderonly, TransformerModel_decoderonly_CTX
from optcar.train import load_checkpoint_weights, load_config


class ForwardWrapper(nn.Module):
    """Export the same named FKD inputs and future_traj output as the controller."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history, history_actions, future_actions, quat_wrt_world,
                vel_wrt_world, angvel_wrt_world, delta_history):
        return self.model({
            "history": history, "history_actions": history_actions,
            "future_actions": future_actions, "quat_wrt_world": quat_wrt_world,
            "vel_wrt_world": vel_wrt_world, "angvel_wrt_world": angvel_wrt_world,
            "delta_history": delta_history,
        })["future_traj"]


def load_model(checkpoint):
    model_cfg = load_config(checkpoint.parent, checkpoint.parent / "model_config.py", ("model_config",))
    learning_cfg = load_config(checkpoint.parent, checkpoint.parent / "learning_config.py", ("learning_config",))
    if not model_cfg.no_map or model_cfg.input_cmd_vel:
        raise ValueError("This MPPI export requires no_map=True and input_cmd_vel=False.")
    model_cfg.device = "cpu"
    models = {"TransformerModel_decoderonly": TransformerModel_decoderonly,
              "TransformerModel_decoderonly_CTX": TransformerModel_decoderonly_CTX}
    if model_cfg.model_type not in models:
        raise ValueError(f"Unsupported model_type: {model_cfg.model_type}")
    model = models[model_cfg.model_type](model_cfg, learning_cfg)
    model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True)
    return model.float().cpu().eval()


def export_onnx(model, path):
    cfg = model.model_config
    inputs = {
        "history": torch.zeros(1, cfg.history_horizon_steps, cfg.input_state_dims),
        "history_actions": torch.zeros(1, cfg.history_horizon_steps, cfg.action_dims),
        "future_actions": torch.zeros(1, cfg.future_horizon_steps, cfg.action_dims),
        "quat_wrt_world": torch.tensor([[1., 0., 0., 0.]]),
        "vel_wrt_world": torch.zeros(1, 3),
        "angvel_wrt_world": torch.zeros(1, 3),
        "delta_history": torch.zeros(1, cfg.history_horizon_steps, cfg.input_state_dims),
    }
    inputs["history"][..., 3] = 1
    inputs["delta_history"][..., 3] = 1
    dynamic_axes = {name: {0: "B"} for name in (*inputs, "future_traj")}
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(
                ForwardWrapper(model).eval(), tuple(inputs.values()), str(path),
                input_names=list(inputs), output_names=["future_traj"],
                dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    import onnx
    onnx.checker.check_model(str(path))
    return {name: list(tensor.shape[1:]) for name, tensor in inputs.items()}


def build_engine(onnx_path, engine_path, shapes, max_batch, precision, workspace_gb):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        name = network.get_input(index).name
        tail = tuple(shapes[name])
        if profile.set_shape(name, (1, *tail), (max_batch, *tail), (max_batch, *tail)) is False:
            raise ValueError(f"Invalid TensorRT input profile: {name}")
    config.add_optimization_profile(profile)
    print(f"Building {precision} TensorRT engine: batch 1–{max_batch} ...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")
    engine_path.write_bytes(bytes(serialized))
    return trt.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", help="Output filename stem; defaults to model_<horizon>h_<max-batch>batch")
    parser.add_argument("--max-batch", type=int, default=601)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-only", action="store_true")
    args = parser.parse_args()
    if min(args.max_batch, args.workspace_gb) < 1:
        parser.error("max-batch and workspace-gb must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model = load_model(checkpoint)
    stem = args.name or f"model_{model.model_config.future_horizon_steps}h_{args.max_batch}batch"
    if Path(stem).name != stem:
        parser.error("name must be a filename stem, not a path")
    onnx_path = checkpoint.parent / f"{stem}.onnx"
    engine_path = checkpoint.parent / f"{stem}.trt"
    if onnx_path.exists() or (not args.onnx_only and engine_path.exists()):
        parser.error("Output already exists; choose a different --name")
    shapes = export_onnx(model, onnx_path)
    print(f"ONNX saved: {onnx_path}", flush=True)
    if args.onnx_only:
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("TensorRT building requires an available CUDA GPU")
    with torch.cuda.device(device):
        version = build_engine(onnx_path, engine_path, shapes, args.max_batch, args.precision, args.workspace_gb)
        metadata = {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_config_sha256": hashlib.sha256((checkpoint.parent / "model_config.py").read_bytes()).hexdigest(),
            "learning_config_sha256": hashlib.sha256((checkpoint.parent / "learning_config.py").read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "history_samples": model.model_config.history_horizon_steps,
            "future_samples": model.model_config.future_horizon_steps,
            "dt": model.model_config.predict_dt,
            "batch_profile": [1, args.max_batch, args.max_batch],
            "precision": args.precision, "io_dtype": "float32",
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "tensorrt": version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "command": [sys.executable, *sys.argv],
        }
    engine_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"TensorRT engine saved: {engine_path}", flush=True)


if __name__ == "__main__":
    main()
