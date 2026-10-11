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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="optcar", description="Offline vehicle dynamics workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create empty data/checkpoint directories")
    init.add_argument("--dataset-root", default="datasets")
    init.add_argument("--checkpoint-root", default="checkpoints")

    gui = commands.add_parser("gui", help="Launch the local web workbench")
    gui.add_argument("--dataset-root", default="datasets")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=5050)
    gui.add_argument("--bag-root", help="Prefill the raw bag conversion folder")

    topics = commands.add_parser("topics", help="List topics in a ROS2 recording")
    topics.add_argument("bag")

    scan = commands.add_parser("scan-bags", help="Discover ROS2 bags and show metadata")
    scan.add_argument("root")
    scan.add_argument("--json", action="store_true", help="Print full machine-readable metadata")
    raw = commands.add_parser("bags-to-bins", help="Convert selected ROS2 bags to raw WheeledLab bins")
    raw.add_argument("root", help="Folder containing bag directories (searched recursively)")
    raw.add_argument("--output", help="Defaults to ROOT's parent / bins")
    raw.add_argument("--select", nargs="+", help="Bag paths relative to ROOT; default: all discovered bags")
    raw.add_argument("--exclude", nargs="+", default=[], help="Relative bag paths to omit")
    raw.add_argument("--config", help="Raw conversion YAML (see configs/raw_bags.yaml)")
    raw.add_argument("--dt", type=float)
    raw.add_argument("--pose-topic")
    raw.add_argument("--twist-topic")
    raw.add_argument("--command-topic")
    raw.add_argument("--aux-twist-topic", help="Optional body twist saved separately; empty name disables auxiliary output")
    raw.add_argument("--twist-frame", choices=["world", "body"])
    raw.add_argument("--command-type", choices=["twist_steering", "ackermann"])
    raw.add_argument("--action-shift", type=float)
    raw.add_argument("--max-duration", type=float)
    raw.add_argument("--include-cmd-vel", action=argparse.BooleanOptionalAction, default=None)
    raw.add_argument("--overwrite", action="store_true", help="Replace only selected bags' bins")

    generate = commands.add_parser("generate", help="Start a synthetic raw-bin collection in tmux")
    generate.add_argument("--config", default="configs/datagen.yaml")
    generate.add_argument("--params", help="DBM YAML; defaults to the terrain's saved preset")
    generate.add_argument("--terrain", required=True)
    generate.add_argument("--dataset-root", default="datasets")
    generate.add_argument("--run", default="default")
    generate.add_argument("--runs", "--episodes", dest="runs", type=int, help="Number of batched runs / bin files (--episodes is a legacy alias)")
    generate.add_argument("--batch-size", type=int)
    generate.add_argument("--device", help="cpu or a visible GPU index, e.g. cuda:0 or cuda:1")
    generate.add_argument("--raw-only", action="store_true", help="Save raw episodes only, for the Dataset builder")

    build = commands.add_parser("build-dataset", help="Build FKD windows from selected real and synthetic raw bins")
    build.add_argument("--config", required=True, help="YAML with sources, data, output, seed and fractions")
    build.add_argument("--device", help="cpu or cuda:N; defaults to the first visible GPU")
    build.add_argument("--workers", type=int, help="Parallel file loading threads (WheeledLab default: 16)")
    build.add_argument("--dataset-root", default="datasets", help="Folder holding the shared tmux job registry")

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
    args = parser.parse_args(argv)
    try:
        _dispatch(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"optcar: {exc}\n")


def _dispatch(args):
    if args.command in ("scan-bags", "bags-to-bins"):
        _raw_bags(args)
    elif args.command == "init":
        for path in (Path(args.dataset_root) / "generalist_dataset", Path(args.dataset_root) / "syn_data/terrains", Path(args.checkpoint_root)):
            path.mkdir(parents=True, exist_ok=True)
        _print({"datasets": args.dataset_root, "checkpoints": args.checkpoint_root})
    elif args.command == "gui":
        from optcar.gui.server import create_app
        create_app(args.dataset_root, args.bag_root).run(host=args.host, port=args.port, threaded=True, debug=False)
    elif args.command == "topics":
        from optcar.datagen.rosbag_to_dataset import topics
        _print(topics(args.bag))
    elif args.command == "generate":
        from optcar.dataset import DataConfig
        from optcar.datagen.dbm_to_dataset import CommandConfig
        from optcar.datagen.generation_jobs import GenerationJobs
        terrain = terrain_path(args.dataset_root, args.terrain)
        target_preset = terrain / "dbm_params.yaml"
        cfg = read_config(args.config)
        preset = read_config(args.params or target_preset)
        if not target_preset.exists():
            write_yaml(target_preset, preset)
        result = GenerationJobs(args.dataset_root).start(args.terrain, args.run,
            data=DataConfig(**cfg.get("data", {})), commands=CommandConfig(**cfg.get("commands", {})),
            runs=args.runs if args.runs is not None else cfg.get("runs", cfg.get("episodes", 10)),
            batch_size=args.batch_size if args.batch_size is not None else cfg.get("batch_size", 32),
            device=args.device or cfg.get("device", "cpu"), preset=preset, fractions=cfg.get("fractions", [.8, .1, .1]))
        _print(result)
    elif args.command == "build-dataset":
        from optcar.dataset import DataConfig
        from optcar.datagen.generation_jobs import GenerationJobs
        cfg = read_config(args.config)
        result = GenerationJobs(args.dataset_root).start_build(cfg["output"], cfg["sources"],
            DataConfig(**cfg.get("data", {})), cfg.get("seed", 42), cfg.get("fractions", [.8, .1, .1]),
            device=args.device or cfg.get("device"), workers=args.workers if args.workers is not None else cfg.get("workers", 16))
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


def _raw_bags(args):
    from optcar.datagen.raw_bags import RawBagConfig, convert_raw_bags, discover_bags
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    console = Console()
    scan = discover_bags(args.root)
    if args.command == "scan-bags":
        if args.json:
            _print(scan)
            return
        table = Table("Bag (relative path)", "Duration", "Size", "Messages", "Metadata")
        for b in scan["bags"]:
            table.add_row(b["relative_path"], f'{b.get("duration_s", 0):.1f}s', f'{b["bytes"] / 2**20:.1f} MB',
                          str(b.get("messages", 0)), b["error"] or "ready")
        console.print(table)
        console.print(f'{len(scan["bags"])} bags · Default output: {scan["default_output"]}', markup=False)
        return
    cfg = read_config(args.config) if args.config else {}
    for key in RawBagConfig.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    config = RawBagConfig(**cfg)
    known = {b["relative_path"]: b["path"] for b in scan["bags"]}
    requested = args.select if args.select is not None else list(known)
    unknown = (set(requested) | set(args.exclude)) - known.keys()
    if unknown:
        raise ValueError(f"Unknown bag paths: {sorted(unknown)}. Use optcar scan-bags ROOT")
    selected = [known[p] for p in requested if p not in args.exclude]
    import signal
    interrupted = False
    def cancel(signum, frame):
        nonlocal interrupted
        interrupted = True
    old_handler = signal.signal(signal.SIGINT, cancel)
    def duration(value):
        return "—" if value is None else f"{value:.0f}s"
    def render(p):
        percent = p.get("percent", 0)
        filled = round(percent / 100 * 30)
        lines = Text()
        lines.append(f"{'━' * filled}{'─' * (30 - filled)} {percent:5.1f}%\n", style="green")
        lines.append(f"Bag {p.get('bag_index', 0)}/{p.get('total_bags', len(selected))} · {p.get('phase', 'starting')}\n")
        lines.append(f"{p.get('bag', '')}\n", style="cyan")
        lines.append(f"Messages {p.get('messages', 0):,}/{p.get('messages_total', 0):,} · Elapsed {duration(p.get('elapsed_s', 0))} · ETA {duration(p.get('eta_s'))}\n")
        rows = p.get("results", [])
        lines.append(f"Saved {sum(r['status'] == 'complete' for r in rows)} · Failed {sum(r['status'] == 'failed' for r in rows)} · Ctrl+C to cancel\n")
        lines.append(f"Output: {p.get('output', args.output or scan['default_output'])}")
        return Panel(lines, title="ROS2 bags → raw bins", border_style="green")
    try:
        with Live(render({}), console=console, refresh_per_second=8) as live:
            result = convert_raw_bags(scan["root"], selected, args.output, config, args.overwrite,
                                      progress=lambda p: live.update(render(p)), cancelled=lambda: interrupted)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    console.print(f"Saved {result['converted']} bags · Failed {result['failed']} · Output: {result['output']}", markup=False)
    for row in result["results"]:
        if row.get("warning"):
            console.print(f"{row['bag']}: {row['warning']}", markup=False, style="yellow")
        if row["status"] == "failed":
            console.print(f"{row['bag']}: {row['error']}", markup=False, style="red")
    if result["cancelled"]:
        raise SystemExit(130)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
