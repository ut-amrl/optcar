"""Bounded one-step DBM identification from observed transitions."""
from dataclasses import asdict
import numpy as np
from scipy.optimize import least_squares
import torch

from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel
from optcar.utils.math import qinverse_torch, qrotate_torch, wrap_to_pi, yaw_from_quat_torch


FIT_BOUNDS = {
    "len_f": (0.01, 5.0), "len_r": (0.01, 5.0), "mass": (0.1, 500.),
    "inertia": (0.001, 500.), "C_tire_f": (0., 5000.), "C_tire_r": (0., 5000.),
    "C_throttle_a": (0.01, 5000.), "C_rollf": (0., 100.), "C_drag": (0., 100.),
    "wheel_time_constant": (0.01, 2.)}
DEFAULT_FIT_KEYS = ["C_tire_f", "C_tire_r", "C_throttle_a", "C_rollf", "C_drag", "wheel_time_constant"]


def wheel_history(commands, initial, tau, dt):
    # Same RK4 integration polynomial as the DBM's first-order actuator.
    z = -dt / tau
    decay = 1 + z + z*z/2 + z*z*z/6 + z*z*z*z/24
    values = np.empty(len(commands), np.float32)
    wheel = initial
    for i, command in enumerate(commands):
        values[i] = wheel
        wheel = decay * wheel + (1 - decay) * command
    return values


def fit_parameters(states, actions, dt, params=None, keys=None, max_evaluations=150, max_samples=5000, progress=None):
    params = params if isinstance(params, DBMParams) else DBMParams(**(params or {}))
    keys = list(keys or DEFAULT_FIT_KEYS)
    if any(key not in FIT_BOUNDS for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Unsupported or duplicated fit parameters")
    if len(states) != len(actions) + 1 or len(actions) < 3:
        raise ValueError("Select at least three continuous transitions for calibration")
    base = asdict(params)
    state = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    command = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    indices = np.linspace(0, len(actions) - 1, min(max_samples, len(actions)), dtype=np.int64)
    current, target = state[:-1][indices], state[1:][indices]
    initial_wheel = float(qrotate_torch(qinverse_torch(state[:1, 3:7]), state[:1, 7:10])[0, 0])
    x0 = np.asarray([base[key] for key in keys], np.float64)
    lower = np.asarray([max(FIT_BOUNDS[key][0], dt / 2) if key == "wheel_time_constant" else FIT_BOUNDS[key][0] for key in keys])
    upper = np.asarray([FIT_BOUNDS[key][1] for key in keys])
    x0 = np.clip(x0, lower + 1e-7, upper - 1e-7)
    evaluations = 0

    @torch.inference_mode()
    def residual(values):
        nonlocal evaluations
        values = {**base, **dict(zip(keys, map(float, values)))}
        model = DynamicBicycleModel(values)
        applied = model.clamp_actions(command)
        wheels = wheel_history(applied[:, 0].numpy(), initial_wheel, values["wheel_time_constant"], dt)
        prediction, _ = model.rollout(current, applied[indices, None], dt, torch.from_numpy(wheels[indices]))
        prediction = prediction[:, 0]
        heading = wrap_to_pi(yaw_from_quat_torch(prediction[:, 3:7]) - yaw_from_quat_torch(target[:, 3:7]))
        errors = torch.cat(((prediction[:, :2] - target[:, :2]) / dt,
                            heading / dt, prediction[:, 7:9] - target[:, 7:9],
                            (prediction[:, 12:13] - target[:, 12:13]) * 0.2), -1)
        evaluations += 1
        if progress and evaluations % 10 == 0:
            progress({"evaluations": evaluations, "rmse": float(errors.square().mean().sqrt())})
        return np.nan_to_num(errors.numpy().astype(np.float64).ravel(), nan=1e6, posinf=1e6, neginf=-1e6)

    def jacobian(values):
        # Absolute steps also identify coefficients initially at zero; SciPy's
        # relative differences there are too small for float32 Torch dynamics.
        columns = []
        for i in range(len(values)):
            step = 1e-3 * max(1.0, abs(values[i]))
            lo, hi = values.copy(), values.copy()
            lo[i] = max(lower[i], values[i] - step)
            hi[i] = min(upper[i], values[i] + step)
            columns.append((residual(hi) - residual(lo)) / (hi[i] - lo[i]))
        return np.stack(columns, axis=1)

    before = residual(x0)
    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_evaluations,
                           jac=jacobian, x_scale="jac", loss="soft_l1")
    fitted = {**base, **dict(zip(keys, map(float, result.x)))}
    return {"params": fitted, "success": bool(result.success), "message": result.message,
            "evaluations": evaluations, "initial_rmse": float(np.sqrt(np.mean(before**2))),
            "fitted_rmse": float(np.sqrt(np.mean(result.fun**2))), "samples": len(indices), "keys": keys}
