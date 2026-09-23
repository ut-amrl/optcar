"""Local Flask workbench adapted from WheeledLab's DBM/FKD viewers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import wraps
from pathlib import Path
import json
import threading
import uuid

from flask import Flask, jsonify, render_template, request
import numpy as np
import torch

from optcar.dataset import DataConfig
from optcar.datagen.rosbag_to_dataset import BagConfig, export_bag, load_bag, topics
from optcar.datagen.dbm_to_dataset import CommandConfig, generate
from optcar.models.dbm_calibration import DEFAULT_FIT_KEYS, fit_parameters, wheel_history
from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel
from optcar.models.fkd_transformer import load_checkpoint, predict_window, trajectory_metrics
from optcar.utils.file_io import read_config, safe_name, terrain_path, write_yaml
from optcar.utils.math import qinverse_torch, qrotate_torch, yaw_from_quat_torch


def create_app(dataset_root="datasets"):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    root = Path(dataset_root).expanduser().resolve()
    (root / "generalist_dataset").mkdir(parents=True, exist_ok=True)
    sessions, jobs = {}, {}
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="optcar")

    def api(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return jsonify(function(*args, **kwargs))
            except (ValueError, TypeError, KeyError, FileNotFoundError, ImportError) as exc:
                return jsonify(error=str(exc)), 400
            except Exception as exc:
                app.logger.exception("OptCar request failed")
                return jsonify(error=str(exc)), 500
        return wrapped

    def get_segment(payload):
        with lock:
            session = sessions.get(payload["session_id"])
        if session is None:
            raise ValueError("Session expired; load the bag again")
        index = int(payload.get("segment", 0))
        if not 0 <= index < len(session["segments"]):
            raise ValueError("Invalid segment")
        return session, session["segments"][index]

    def start_job(operation):
        identifier = uuid.uuid4().hex
        with lock:
            if sum(j["status"] in ("queued", "running") for j in jobs.values()) >= 2:
                raise ValueError("Two jobs are already active; wait or cancel one")
            # Keep only a bounded number of completed job records.
            for key in list(jobs):
                if len(jobs) <= 32:
                    break
                if jobs[key]["status"] not in ("queued", "running"):
                    del jobs[key]
            jobs[identifier] = {"status": "queued", "progress": {}, "cancel": False}
        def progress(value):
            with lock:
                jobs[identifier]["progress"] = value
        def cancelled():
            with lock:
                return jobs[identifier]["cancel"]
        def work():
            with lock:
                jobs[identifier]["status"] = "running"
            try:
                result = operation(progress, cancelled)
                with lock:
                    jobs[identifier].update(status="cancelled" if cancelled() else "complete", result=result)
            except Exception as exc:
                app.logger.exception("OptCar job failed")
                with lock:
                    jobs[identifier].update(status="failed", error=str(exc))
        pool.submit(work)
        return {"job_id": identifier}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/defaults")
    @api
    def defaults():
        return {"params": asdict(DBMParams()), "data": asdict(DataConfig()), "commands": asdict(CommandConfig()),
                "bag": asdict(BagConfig()), "fit_keys": DEFAULT_FIT_KEYS, "dataset_root": str(root)}

    @app.post("/api/topics")
    @api
    def list_topics():
        return {"topics": topics(request.get_json()["path"])}

    @app.post("/api/bag")
    @api
    def open_bag():
        payload = request.get_json()
        split = payload.get("split", "train")
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid bag split")
        session = load_bag(payload["path"], BagConfig(**payload.get("bag", {})), float(payload.get("dt", 0.02)))
        session["split"] = split
        identifier = uuid.uuid4().hex
        with lock:
            if len(sessions) >= 4:
                sessions.pop(next(iter(sessions)))
            sessions[identifier] = session
        return {"session_id": identifier, "dt": session["dt"], "split": split,
                "segments": [{"index": i, "samples": len(s["states"]), "start_s": float(s["times"][0]),
                              "end_s": float(s["times"][-1])} for i, s in enumerate(session["segments"])]}

    @app.post("/api/segment")
    @api
    def segment_data():
        session, segment = get_segment(request.get_json())
        stride = max(1, int(np.ceil(len(segment["states"]) / 6000)))
        indices = np.arange(0, len(segment["states"]), stride)
        state = segment["states"][indices]
        commands = segment["actions"][np.minimum(indices, len(segment["actions"]) - 1)]
        return {"states": state.tolist(), "commands": commands.tolist(), "indices": indices.tolist(),
                "times": segment["times"][indices].tolist(), "samples": len(segment["states"]),
                "yaw": yaw_from_quat_torch(torch.from_numpy(state[:, 3:7])).squeeze(-1).tolist(), "dt": session["dt"]}

    @app.post("/api/rollout")
    @api
    def rollout():
        payload = request.get_json()
        session, segment = get_segment(payload)
        current, horizon = int(payload["current"]), int(payload["horizon"])
        if current < 0 or horizon < 1 or current + horizon >= len(segment["states"]):
            raise ValueError("Prediction window extends beyond this segment")
        model = DynamicBicycleModel(payload["params"])
        initial = torch.from_numpy(segment["states"][current:current + 1])
        actions = torch.from_numpy(segment["actions"][current:current + horizon])[None]
        wheel = payload.get("wheel_speed")
        if wheel is None:
            first = torch.from_numpy(segment["states"][:1])
            initial_wheel = float(qrotate_torch(qinverse_torch(first[:, 3:7]), first[:, 7:10])[0, 0])
            recorded = model.clamp_actions(torch.from_numpy(segment["actions"][:current + 1]))[:, 0].numpy()
            wheel = float(wheel_history(recorded, initial_wheel, model.params.wheel_time_constant, session["dt"])[-1])
        with torch.inference_mode():
            predicted, _ = model.rollout(initial, actions, session["dt"], [float(wheel)])
        poses = predicted[0, :, :7].numpy()
        measured = segment["states"][current + 1:current + horizon + 1, :7]
        return {"predicted": poses.tolist(), "measured": measured.tolist(), "current": current,
                "horizon": horizon, "metrics": trajectory_metrics(poses, measured)}

    @app.post("/api/fit")
    @api
    def fit():
        payload = request.get_json()
        session, segment = get_segment(payload)
        if session["split"] != "train":
            raise ValueError("Fit only training bags; use validation/test bags for preview")
        start, end = int(payload["start"]), int(payload["end"])
        if start < 0 or end >= len(segment["states"]) or end - start < 3:
            raise ValueError("Invalid calibration interval")
        def operation(progress, cancelled):
            def update(value):
                if cancelled():
                    raise ValueError("Calibration cancelled")
                progress(value)
            result = fit_parameters(segment["states"][start:end + 1], segment["actions"][start:end],
                                    session["dt"], payload["params"], payload.get("keys"), progress=update)
            provenance = {"session": session["session"], "path": session["path"], "split": "train",
                          "segment": int(payload.get("segment", 0)), "start": start, "end": end,
                          "bag_config": session["config"], "fit": {k: v for k, v in result.items() if k != "params"}}
            with lock:
                session["calibration"] = provenance
            return {**result, "calibration": provenance}
        return start_job(operation)

    @app.route("/api/preset", methods=["GET", "POST"])
    @api
    def preset():
        if request.method == "GET":
            path = terrain_path(root, request.args["terrain"]) / "dbm_params.yaml"
            return read_config(path)
        payload = request.get_json()
        params = asdict(DBMParams(**payload["params"]))
        path = terrain_path(root, payload["terrain"]) / "dbm_params.yaml"
        provenance = payload.get("calibration") or {}
        if not provenance and path.exists():
            provenance = read_config(path).get("calibration", {})
        if payload.get("session_id"):
            session, _ = get_segment(payload)
            provenance = session.get("calibration") or provenance
        if provenance and provenance.get("split") != "train":
            raise ValueError("Calibration provenance must identify training data")
        write_yaml(path, {"params": params, "calibration": provenance})
        return {"path": str(path), "params": params}

    @app.post("/api/export")
    @api
    def export():
        payload = request.get_json()
        session, _ = get_segment(payload)
        path = terrain_path(root, payload["terrain"]) / "real" / safe_name(payload.get("run", "default"))
        data = DataConfig(**payload.get("data", {}))
        return start_job(lambda progress, cancelled: export_bag(session, path, payload["terrain"], data, session["split"]))

    @app.post("/api/generate")
    @api
    def generate_data():
        payload = request.get_json()
        terrain = terrain_path(root, payload["terrain"])
        preset = read_config(terrain / "dbm_params.yaml")
        path = terrain / "sim" / safe_name(payload.get("run", "default"))
        return start_job(lambda progress, cancelled: generate(path, payload["terrain"], preset["params"],
            DataConfig(**payload.get("data", {})), CommandConfig(**payload.get("commands", {})),
            episodes=int(payload.get("episodes", 1000)), batch_size=int(payload.get("batch_size", 128)),
            device=payload.get("device", "cpu"), progress=progress, cancelled=cancelled,
            calibration=preset.get("calibration")))

    @app.get("/api/jobs/<identifier>")
    @api
    def job_status(identifier):
        with lock:
            return dict(jobs[identifier])

    @app.post("/api/jobs/<identifier>/cancel")
    @api
    def cancel_job(identifier):
        with lock:
            jobs[identifier]["cancel"] = True
        return {"status": "cancellation requested"}

    @app.get("/api/datasets")
    @api
    def datasets():
        results = []
        for path in sorted(root.glob("syn_data/terrains/*/*/*/manifest.json")):
            doc = json.loads(path.read_text())
            counts = {split: sum(e["windows"] for e in doc["episodes"].values() if e["split"] == split)
                      for split in ("train", "val", "test")}
            results.append({"path": str(path.parent.relative_to(root)), "terrain": doc["terrain"],
                            "source": doc["source"], "windows": counts})
        return {"datasets": results}

    @app.post("/api/checkpoint")
    @api
    def checkpoint_info():
        model, data, checkpoint = load_checkpoint(request.get_json()["path"])
        return {"model": asdict(model.config), "data": asdict(data), "stage": checkpoint["stage"],
                "epoch": checkpoint["epoch"], "metrics": checkpoint.get("metrics", {})}

    @app.post("/api/predict")
    @api
    def predict():
        payload = request.get_json()
        session, segment = get_segment(payload)
        model, config, _ = load_checkpoint(payload["checkpoint"], payload.get("device", "cpu"))
        if abs(config.dt - session["dt"]) > 1e-9:
            raise ValueError(f"Reload bag at checkpoint timestep {config.dt}")
        return predict_window(model, config, segment["states"], segment["actions"],
                              int(payload["current"]), int(payload["horizon"]))

    return app
