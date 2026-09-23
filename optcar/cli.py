"""Command-line entrypoints for the complete offline workflow."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from optcar.utils.file_io import read_config, safe_name, terrain_path, write_json, write_yaml


def _print(value):
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)


def _bag_config(path):
    from optcar.datagen.rosbag_to_dataset import BagConfig
    return BagConfig(**(read_config(path) if path else {}))


def main():
    parser = argparse.ArgumentParser(prog="optcar", description="Offline vehicle dynamics workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create empty data/checkpoint directories")
    init.add_argument("--dataset-root", default="datasets")
    init.add_argument("--checkpoint-root", default="checkpoints")

    gui = commands.add_parser("gui", help="Launch the local web workbench")
    gui.add_argument("--dataset-root", default="datasets")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=5050)

    topics = commands.add_parser("topics", help="List topics in a ROS2 recording")
    topics.add_argument("bag")

    generate = commands.add_parser("generate", help="Generate DBM training data")
    generate.add_argument("--config", default="configs/datagen.yaml")
    generate.add_argument("--params", help="DBM YAML; defaults to the terrain's saved preset")
    generate.add_argument("--terrain", required=True)
    generate.add_argument("--dataset-root", default="datasets")
    generate.add_argument("--run", default="default")
    generate.add_argument("--episodes", type=int)
    generate.add_argument("--batch-size", type=int)
    generate.add_argument("--device")

    convert = commands.add_parser("convert", help="Convert a complete bag to FKD bins")
    convert.add_argument("bag")
    convert.add_argument("--bag-config", default="configs/rosbag.yaml")
    convert.add_argument("--config", default="configs/datagen.yaml")
    convert.add_argument("--terrain", required=True)
    convert.add_argument("--split", choices=["train", "val", "test"], required=True)
    convert.add_argument("--dataset-root", default="datasets")
    convert.add_argument("--run", default="default")
    convert.add_argument("--vehicle-id", type=int, default=0)

    fit = commands.add_parser("fit", help="Identify DBM parameters on a training bag segment")
    fit.add_argument("bag")
    fit.add_argument("--bag-config", default="configs/rosbag.yaml")
    fit.add_argument("--params", default="configs/dbm.yaml")
    fit_target = fit.add_mutually_exclusive_group(required=True)
    fit_target.add_argument("--terrain", help="Save the fitted preset for this terrain")
    fit_target.add_argument("--output", help="Save the fitted preset to a specific YAML path")
    fit.add_argument("--dataset-root", default="datasets")
    fit.add_argument("--segment", type=int, default=0)
    fit.add_argument("--start", type=int, default=0)
    fit.add_argument("--end", type=int)
    fit.add_argument("--dt", type=float, default=0.02)
    fit.add_argument("--keys", nargs="+")
    fit.add_argument("--max-evaluations", type=int, default=150)

    train = commands.add_parser("train", help="Train a generalist or fine-tune a specialist")
    train.add_argument("--config", required=True)
    train.add_argument("--architecture", choices=["film", "cross_attention"])
    train.add_argument("--checkpoint")
    train.add_argument("--resume")
    train.add_argument("--device")
    train.add_argument("--output")

    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on held-out bins")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--datasets", nargs="+", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--real-only", action="store_true")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--output")

    predict = commands.add_parser("predict", help="Predict one recorded window from a checkpoint")
    predict.add_argument("bag")
    predict.add_argument("--bag-config", default="configs/rosbag.yaml")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--segment", type=int, default=0)
    predict.add_argument("--current", type=int, required=True)
    predict.add_argument("--horizon", type=int)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        _dispatch(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"optcar: {exc}\n")


def _dispatch(args):
    if args.command == "init":
        for path in (Path(args.dataset_root) / "generalist_dataset", Path(args.dataset_root) / "syn_data/terrains", Path(args.checkpoint_root)):
            path.mkdir(parents=True, exist_ok=True)
        _print({"datasets": args.dataset_root, "checkpoints": args.checkpoint_root})
    elif args.command == "gui":
        from optcar.gui.server import create_app
        create_app(args.dataset_root).run(host=args.host, port=args.port, threaded=True, debug=False)
    elif args.command == "topics":
        from optcar.datagen.rosbag_to_dataset import topics
        _print(topics(args.bag))
    elif args.command == "generate":
        from optcar.dataset import DataConfig
        from optcar.datagen.dbm_to_dataset import CommandConfig, generate
        terrain = terrain_path(args.dataset_root, args.terrain)
        target_preset = terrain / "dbm_params.yaml"
        cfg = read_config(args.config)
        preset = read_config(args.params or target_preset)
        if not target_preset.exists():
            write_yaml(target_preset, preset)
        result = generate(terrain / "sim" / safe_name(args.run), args.terrain, preset.get("params", preset),
            DataConfig(**cfg.get("data", {})), CommandConfig(**cfg.get("commands", {})),
            episodes=args.episodes if args.episodes is not None else cfg.get("episodes", 1000),
            batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 128),
            device=args.device or cfg.get("device", "cpu"), fractions=cfg.get("fractions", [0.8, 0.1, 0.1]),
            progress=lambda p: print(f"Episodes {p['completed']}/{p['total']} · new windows {p['new_windows']}", flush=True)
                if p["completed"] % 100 == 0 or p["completed"] == p["total"] else None,
            calibration=preset.get("calibration"))
        _print(result)
    elif args.command == "convert":
        from optcar.dataset import DataConfig
        from optcar.datagen.rosbag_to_dataset import export_bag, load_bag
        data = DataConfig(**read_config(args.config).get("data", {}))
        session = load_bag(args.bag, _bag_config(args.bag_config), data.dt)
        root = terrain_path(args.dataset_root, args.terrain) / "real" / safe_name(args.run)
        _print(export_bag(session, root, args.terrain, data, args.split, args.vehicle_id))
    elif args.command == "fit":
        from optcar.datagen.rosbag_to_dataset import load_bag
        from optcar.models.dbm_calibration import fit_parameters
        session = load_bag(args.bag, _bag_config(args.bag_config), args.dt)
        if not 0 <= args.segment < len(session["segments"]):
            raise ValueError("Invalid continuous segment index")
        segment = session["segments"][args.segment]
        end = args.end if args.end is not None else len(segment["states"]) - 1
        if args.start < 0 or end >= len(segment["states"]) or end - args.start < 3:
            raise ValueError("Invalid fit interval")
        preset = read_config(args.params)
        result = fit_parameters(segment["states"][args.start:end + 1], segment["actions"][args.start:end],
            args.dt, preset.get("params", preset), args.keys, args.max_evaluations, progress=_print)
        calibration = {"session": session["session"], "path": session["path"], "split": "train",
                       "segment": args.segment, "start": args.start, "end": end, "bag_config": session["config"],
                       "fit": {k: v for k, v in result.items() if k != "params"}}
        output = args.output or terrain_path(args.dataset_root, args.terrain) / "dbm_params.yaml"
        write_yaml(output, {"params": result["params"], "calibration": calibration})
        _print(result)
    elif args.command == "train":
        from optcar.dataset import DataConfig
        from optcar.models.fkd_transformer import ModelConfig
        from optcar.train import TrainConfig, train
        cfg = read_config(args.config)
        training, model = cfg.get("training", {}), cfg.get("model", {})
        for key in ("checkpoint", "resume", "device", "output"):
            if getattr(args, key) is not None:
                training[key] = getattr(args, key)
        if args.architecture:
            model["architecture"] = args.architecture
        _print(train(TrainConfig(**training), DataConfig(**cfg.get("data", {})), ModelConfig(**model), progress=_print))
    elif args.command == "evaluate":
        from optcar.evaluate import evaluate
        result = evaluate(args.checkpoint, args.datasets, args.split, args.device, args.batch_size, args.real_only)
        if args.output:
            write_json(args.output, result)
        _print(result)
    elif args.command == "predict":
        from optcar.datagen.rosbag_to_dataset import load_bag
        from optcar.models.fkd_transformer import load_checkpoint
        from optcar.models.fkd_transformer import predict_window
        model, data, _ = load_checkpoint(args.checkpoint, args.device)
        session = load_bag(args.bag, _bag_config(args.bag_config), data.dt)
        if not 0 <= args.segment < len(session["segments"]):
            raise ValueError("Invalid continuous segment index")
        segment = session["segments"][args.segment]
        result = predict_window(model, data, segment["states"], segment["actions"], args.current, args.horizon)
        write_json(args.output, result)
        _print(result["metrics"])


if __name__ == "__main__":
    main()
