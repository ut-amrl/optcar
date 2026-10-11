"""Durable tmux jobs around the synthetic collector and dataset builder."""
from dataclasses import asdict
from pathlib import Path
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from optcar.utils.file_io import read_config, safe_name, terrain_path, write_json

ACTIVE = {"queued", "running", "cancelling"}
PROJECT = Path(__file__).resolve().parents[2]


class GenerationJobs:
    def __init__(self, dataset_root):
        self.root = Path(dataset_root).expanduser().resolve()
        self.directory = self.root / ".generation_jobs"

    def job_dir(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid generation job ID")
        return self.directory / identifier

    def exists(self, identifier):
        return bool(re.fullmatch(r"[0-9a-f]{32}", identifier)) and (self.job_dir(identifier) / "status.json").exists()

    def _panes(self):
        if not shutil.which("tmux"):
            return {}
        result = subprocess.run(["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{pane_dead}\t#{pane_dead_status}"],
                                capture_output=True, text=True, timeout=5)
        return {parts[0]: parts[1:] for line in result.stdout.splitlines()
                if len(parts := line.split("\t")) == 3}

    def _read(self, identifier, panes):
        directory = self.job_dir(identifier)
        record = json.loads((directory / "status.json").read_text())
        pane = panes.get(record["tmux_session"])
        record["session_available"] = pane is not None
        # Give a newly launched worker time to start before treating it as lost.
        if record["status"] in ACTIVE and time.time() - record["created_at"] > 15 and (pane is None or pane[0] == "1"):
            # Re-read before writing: the worker may have completed during the tmux query.
            record = json.loads((directory / "status.json").read_text())
            if record["status"] in ACTIVE:
                record.update(status="failed", finished_at=time.time(),
                              error="Worker stopped before reporting completion. Inspect the tmux session." if pane else
                                    "The tmux session was closed before the job finished. Completed bins are retained.")
                write_json(directory / "status.json", record)
            record["session_available"] = pane is not None
        if (directory / "cancel").exists() and record["status"] in ACTIVE:
            record["status"] = "cancelling"
        return record

    def get(self, identifier):
        return self._read(identifier, self._panes())

    def list(self, kind=None):
        panes = self._panes()
        records = [self._read(path.parent.name, panes) for path in self.directory.glob("*/status.json")]
        return sorted((r for r in records if kind is None or r.get("kind", "generation") == kind),
                      key=lambda item: item["created_at"], reverse=True)

    def cancel(self, identifier):
        record = self.get(identifier)
        if record["status"] in ACTIVE:
            (self.job_dir(identifier) / "cancel").touch()
        return {"status": "cancellation requested" if record["status"] in ACTIVE else record["status"]}

    def start(self, terrain, run="default", data=None, commands=None, runs=10, batch_size=32,
              device="cpu", preset=None, fractions=(.8, .1, .1)):
        from optcar.dataset import DataConfig
        from optcar.datagen.dbm_to_dataset import CommandConfig, randomized_params, simulation_params
        from optcar.utils.devices import resolve_device
        tmux = shutil.which("tmux")
        if tmux is None:
            raise ValueError("tmux is required for generation. Install tmux, then restart the GUI.")
        terrain, run = safe_name(terrain), safe_name(run)
        data = data if isinstance(data, DataConfig) else DataConfig(**(data or {}))
        commands = commands if isinstance(commands, CommandConfig) else CommandConfig(**(commands or {}))
        if any(type(value) is not int or value < 1 for value in (runs, batch_size)):
            raise ValueError("Runs and batch size must be positive integers")
        steps = round(commands.duration / data.dt)
        if steps < 1 or abs(steps * data.dt - commands.duration) > 1e-6:
            raise ValueError("Duration must be an integer multiple of dt")
        if len(fractions) != 3 or any(not math.isfinite(v) or v < 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-6:
            raise ValueError("Three split fractions must sum to one")
        device = resolve_device(device)
        preset = preset or read_config(terrain_path(self.root, terrain) / "dbm_params.yaml")
        params = preset.get("params", preset)
        randomized_params(simulation_params(params, commands, data.dt), commands, 0)
        output = (Path(preset["path"]).expanduser().resolve().parent / run if preset.get("path") else
                  terrain_path(self.root, terrain) / "sim" / run)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if any(record["status"] in ACTIVE and record["dataset"] == str(output) for record in self.list()):
                raise ValueError("This output already has an active collection. Choose a different run name.")
            identifier = uuid.uuid4().hex
            directory = self.job_dir(identifier)
            directory.mkdir()
            session = f"optcar-gen-{terrain[:20]}-{run[:24]}-{identifier[:8]}".replace(".", "-")
            record = {"job_id": identifier, "kind": "generation", "status": "queued", "terrain": terrain,
                      "run": run, "device": device, "dataset": str(output), "tmux_session": session,
                      "preset_path": preset.get("path"),
                      "attach_command": shlex.join(["tmux", "attach", "-t", session]), "created_at": time.time(),
                      "runs": runs, "batch_size": batch_size, "duration": commands.duration, "dt": data.dt,
                      "progress": {"completed": 0, "total": runs * steps}}
            write_json(directory / "request.json", {"params": params, "calibration": preset.get("calibration"),
                       "preset_path": preset.get("path"),
                       "data": asdict(data), "commands": asdict(commands), "fractions": list(fractions)})
            write_json(directory / "status.json", record)
            self._launch(directory, record, tmux)
        return {"job_id": identifier, "tmux_session": session, "attach_command": record["attach_command"], "dataset": str(output)}


    def _launch(self, directory, record, tmux):
        # Use the GUI/CLI's exact interpreter and GPU visibility, rather than the tmux server's old environment.
        env_args = ["env", "-u", "CUDA_VISIBLE_DEVICES", "-u", "NVIDIA_VISIBLE_DEVICES"]
        env_args += [f"{key}={os.environ[key]}" for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES") if key in os.environ]
        worker = env_args + [sys.executable, "-u", str(PROJECT / "scripts/generation_worker.py"), str(directory)]
        shell_command = shlex.join([tmux, "set-option", "-w", "remain-on-exit", "on"]) + " && exec " + shlex.join(worker)
        try:
            subprocess.run([tmux, "new-session", "-d", "-s", record["tmux_session"], "-c", str(PROJECT), shell_command],
                           check=True, capture_output=True, text=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as exc:
            error = getattr(exc, "stderr", None) or str(exc)
            record.update(status="failed", error=f"Could not start tmux: {error.strip()}", finished_at=time.time())
            write_json(directory / "status.json", record)
            raise ValueError(record["error"]) from exc

    def start_build(self, output, sources, data=None, seed=42, fractions=(.8, .1, .1), device=None, workers=16):
        from optcar.dataset import DataConfig, session_split
        from optcar.datagen.build_dataset import build_output
        from optcar.utils.devices import available_devices, resolve_device
        tmux = shutil.which("tmux")
        if not tmux:
            raise ValueError("tmux is required for dataset building. Install tmux, then restart the GUI.")
        data = data if isinstance(data, DataConfig) else DataConfig(**(data or {}))
        if type(workers) is not int or workers < 1:
            raise ValueError("File loading workers must be a positive integer")
        device = resolve_device(device if device is not None else available_devices()["default"])
        if not sources or not str(output).strip():
            raise ValueError("Choose an output folder and at least one raw bin source")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Split seed must be a nonnegative integer")
        session_split("validate", seed, fractions)
        target = build_output(output, sources)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if any(record["status"] in ACTIVE and record["dataset"] == str(target) for record in self.list()):
                raise ValueError("This output already has an active job. Choose a different output folder.")
            identifier = uuid.uuid4().hex
            directory = self.job_dir(identifier)
            directory.mkdir()
            session = f"optcar-build-{identifier[:8]}"
            record = {"job_id": identifier, "kind": "dataset_build", "status": "queued", "device": device, "workers": workers,
                      "dataset": str(target), "tmux_session": session, "created_at": time.time(),
                      "attach_command": shlex.join(["tmux", "attach", "-t", session]),
                      "progress": {"completed": 0, "total": 0, "new_windows": 0, "current": "Starting worker"}}
            write_json(directory / "request.json", {"output": str(Path(output).expanduser().resolve()),
                       "sources": sources, "data": asdict(data), "seed": seed, "fractions": list(fractions), "workers": workers})
            write_json(directory / "status.json", record)
            self._launch(directory, record, tmux)
        return {"job_id": identifier, "tmux_session": session, "attach_command": record["attach_command"], "dataset": str(target)}


def run_worker(directory):
    """Same collector as before; progress is shared with tmux and the web dashboard."""
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from optcar.dataset import DataConfig
    from optcar.datagen.dbm_to_dataset import CommandConfig, generate
    directory = Path(directory)
    record = json.loads((directory / "status.json").read_text())
    config = json.loads((directory / "request.json").read_text())
    console = Console()
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def cancelled():
        return stopped or (directory / "cancel").exists()
    record.update(status="running", started_at=time.time(), pid=os.getpid())
    write_json(directory / "status.json", record)
    label = 'Dataset build' if record['kind'] == 'dataset_build' else f"{record['terrain']} / {record['run']}"
    console.print(f"OptCar · {label} · {record['device']}", markup=False)
    if record["kind"] == "dataset_build":
        console.print(f"File loading workers: {config.get('workers', 16)}", markup=False)
    console.print(f"Session: {record['tmux_session']}\nOutput: {record['dataset']}\nCtrl+C cancels; Ctrl+B then D detaches.", markup=False)
    last_saved = 0.0
    try:
        with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                      TimeElapsedColumn(), TimeRemainingColumn(), console=console, refresh_per_second=4) as live:
            task = live.add_task("Starting…", total=record["progress"]["total"] or None)
            def update(value):
                nonlocal last_saved
                record["progress"] = value
                now = time.monotonic()
                if now - last_saved >= .5 or value["completed"] == value["total"]:
                    record["updated_at"] = time.time()
                    write_json(directory / "status.json", record)
                    live.update(task, completed=value["completed"], total=value["total"] or None, description=value["current"])
                    last_saved = now
            if record["kind"] == "dataset_build":
                from optcar.datagen.build_dataset import build_dataset
                result = build_dataset(config["output"], config["sources"], DataConfig(**config["data"]),
                                       config["seed"], config["fractions"], update, cancelled, device=record["device"],
                                       workers=config.get("workers", 16))
            else:
                result = generate(record["dataset"], record["terrain"], config["params"],
                                  DataConfig(**config["data"]), CommandConfig(**config["commands"]),
                                  runs=record["runs"], batch_size=record["batch_size"], device=record["device"],
                                  fractions=config["fractions"], progress=update, cancelled=cancelled,
                                  calibration=config.get("calibration"), process_windows=False)
            complete = result["completed"] == result["total"]
            record.update(status="complete" if complete else "cancelled", result=result)
            if complete:
                record["progress"]["completed"] = record["progress"]["total"]
                live.update(task, completed=record["progress"]["total"], description="Complete")
        count = f"{result['new_windows']:,} new windows" if record["kind"] == "dataset_build" else f"{result['trajectories']} trajectories"
        console.print(f"{record['status'].capitalize()} · {result['completed']} bins · {count}", markup=False)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        console.print(traceback.format_exc(), markup=False)
    finally:
        record["finished_at"] = time.time()
        write_json(directory / "status.json", record)
    return 1 if record["status"] == "failed" else 0
