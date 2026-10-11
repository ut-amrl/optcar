"""Durable tmux jobs around the existing synthetic collector."""
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
                              error="Generation worker stopped before reporting completion. Inspect the tmux session." if pane else
                                    "The tmux session was closed before generation finished. Completed bins are retained.")
                write_json(directory / "status.json", record)
            record["session_available"] = pane is not None
        if (directory / "cancel").exists() and record["status"] in ACTIVE:
            record["status"] = "cancelling"
        return record

    def get(self, identifier):
        return self._read(identifier, self._panes())

    def list(self):
        panes = self._panes()
        records = [self._read(path.parent.name, panes) for path in self.directory.glob("*/status.json")]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

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
            # Use the GUI/CLI's exact interpreter and GPU visibility, rather than the tmux server's old environment.
            env_args = ["env", "-u", "CUDA_VISIBLE_DEVICES", "-u", "NVIDIA_VISIBLE_DEVICES"]
            env_args += [f"{key}={os.environ[key]}" for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES") if key in os.environ]
            worker = env_args + [sys.executable, "-u", str(PROJECT / "scripts/generation_worker.py"), str(directory)]
            shell_command = shlex.join([tmux, "set-option", "-w", "remain-on-exit", "on"]) + " && exec " + shlex.join(worker)
            try:
                subprocess.run([tmux, "new-session", "-d", "-s", session, "-c", str(PROJECT), shell_command],
                               check=True, capture_output=True, text=True, timeout=10)
            except (subprocess.SubprocessError, OSError) as exc:
                error = getattr(exc, "stderr", None) or str(exc)
                record.update(status="failed", error=f"Could not start tmux: {error.strip()}", finished_at=time.time())
                write_json(directory / "status.json", record)
                raise ValueError(record["error"]) from exc
        return {"job_id": identifier, "tmux_session": session, "attach_command": record["attach_command"], "dataset": str(output)}


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
    console.print(f"OptCar · {record['terrain']} / {record['run']} · {record['device']}", markup=False)
    console.print(f"Session: {record['tmux_session']}\nOutput: {record['dataset']}\nCtrl+C cancels; Ctrl+B then D detaches.", markup=False)
    last_saved = 0.0
    try:
        with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                      TimeElapsedColumn(), TimeRemainingColumn(), console=console, refresh_per_second=4) as live:
            task = live.add_task("Starting…", total=record["progress"]["total"])
            def update(value):
                nonlocal last_saved
                record["progress"] = value
                now = time.monotonic()
                if now - last_saved >= .5 or value["completed"] == value["total"]:
                    record["updated_at"] = time.time()
                    write_json(directory / "status.json", record)
                    live.update(task, completed=value["completed"], total=value["total"], description=value["current"])
                    last_saved = now
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
        console.print(f"{record['status'].capitalize()} · {result['completed']} bins · {result['trajectories']} trajectories", markup=False)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        console.print(traceback.format_exc(), markup=False)
    finally:
        record["finished_at"] = time.time()
        write_json(directory / "status.json", record)
    return 1 if record["status"] == "failed" else 0
