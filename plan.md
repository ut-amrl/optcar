# OptCar implementation plan

Build an offline pipeline: ROS2 bags → terrain-specific DBM calibration → synthetic data → generalist/specialist training → `.ckpt` → trajectory visualization.

Reference: [OptCar paper](https://arxiv.org/html/2607.13319v1), Sections 4–5 and Appendix A. This phase delivers data generation, training, and visualization; MPC integration is outside scope.

## 1. Package and reuse

Keep the implementation inside `source/optcar`, with an installable `optcar` Python package:

```text
optcar/
  models/       # Dynamic bicycle model, calibration, FKD transformer + checkpoints
  datagen/      # ROS2 conversion and DBM environment/command generation
  dataset.py    # Binary format, preprocessing, writer, loader, sampling
  train.py      # Generalist training and specialist fine-tuning
  evaluate.py   # Prediction losses and held-out evaluation
  gui/          # Web server, templates, static assets
  utils/        # Copied math utilities and frame transforms
configs/        # DBM/terrain, data generation, model, training presets
datasets/
checkpoints/
```

- Port the Torch DBM from `wheeledlab_models/analytical/bicycle/dynamic_bicycle_model.py` and adapt `wheeledlab_envs/analytical/DynamicBicycleEnv` with its required base-environment logic.
- Use Torch on both CPU and CUDA. The existing DBM imports CuPy and selects it on CUDA; remove that dependency and dispatch from the OptCar port.
- Reuse `ParametricOpenLoopController`, FKD preprocessing/dataloader/training logic, and existing transformer implementations where applicable.
- Copy `wheeledlab_utils/math.py` into `optcar/utils/math.py`; use it consistently for quaternion, angle, and frame operations. Bring over required transform helpers.
- Adapt the existing Flask DBM estimator (`wheeledlab_utils/dbm_estimator`) and FKD viewer (`wheeledlab_learning/tools/FKD`) for the web interface.

## 2. Dataset structure and binary format

```text
datasets/
  generalist_dataset/                 # Empty until data is supplied
  syn_data/
    terrains/
      <terrain>/
        dbm_params.yaml
        sim/                         # DBM-generated data
        real/                        # ROS2-bag-derived data
```

Each populated `sim/` or `real/` contains `raw/`, `processed/data/*.bin`, FKD-compatible metadata, and a manifest recording source sessions, splits, seeds, timestep, and parameter presets. Both sources use the same writer and preprocessing path.

- Preserve FKD float32 binary layout: world state (13), body state (13), delta state (13), commands (2), vehicle ID (1): **42 features** per timestep.
- Commands mean velocity in m/s and steering angle in radians. Use scalar-first quaternions and explicit world/body-frame conventions.
- Record shard shapes, history/future lengths, and feature order; preserve the existing loader’s current-state indexing and verify compatibility with a round trip.
- Split by complete bag/session or simulated episode **before** extracting overlapping windows. Use training data only for normalization and DBM fitting.
- Keep `generalist_dataset/` genuinely empty; create it during setup without placeholder data.

## 3. Two data-generation sources

**ROS2 bags:** load bag directories or supported MCAP/SQLite recordings; configure pose, twist, and command topics; align timestamps and resample to the control timestep. Split at recording gaps, normalize quaternions, and export raw and processed bins. Keep ROS dependencies optional for synthetic generation and training.

**DBM simulation:** batch independent Torch environments on CPU/GPU, using a terrain’s saved DBM parameters. Preserve actuator/wheel state across steps and reset it between episodes. Stream bounded-size shards for large datasets; support restart and reproducible, distinct episode seeds.

Default command preset, matching the requested mission:

| Setting | Default |
|---|---|
| Duration / control timestep | 10 s / 0.02 s = 500 steps |
| Velocity | Bézier, bounds `[-2, 5]` m/s |
| Steering | Random walk, bounds `[-0.5, 0.5]` rad |
| Noise standard deviation | `(0.2, 0.0)`, configurable per channel |
| Offset fraction / base seed | `0.0` / `42` |

Apply bounds after noise and save the actual applied commands. Expose episode count, batch size, device, initial-state ranges, and optional parameter randomization around calibrated presets. The simulator serves offline data collection and calibration previews only.

## 4. Transformers and two-stage training

Provide two selectable architectures with the same input/output contract:

- **FiLM:** encode recent state/action history into a context vector that modulates every rollout block; preserve the paper’s history-context bottleneck.
- **Cross-attention:** future-command decoder attends to encoded history tokens, providing a comparable alternative.

Both consume history, current state, and future commands; predict body-frame pose increments; and reconstruct world trajectories using shared math utilities. Exclude future measured states from model inputs. Start with configurable defaults of 250 history steps, 50 future steps, embedding size 64, two history/decoder layers, and four attention heads.

1. **Generalist:** train the selected architecture on `generalist_dataset/`; save `generalist_best.ckpt`. If empty, report that data is required.
2. **Specialist:** initialize from a matching generalist checkpoint and fine-tune on all selected terrains’ `real/` + `sim/` bins. Configure real/synthetic sampling weights and terrain balance; save `specialist_best.ckpt`.

Use position/quaternion prediction losses, AdamW, warmup, validation, resume support, and configurable training budgets. Each checkpoint includes weights, architecture, normalization, timestep, feature conventions, and training state. Report held-out real-data position/heading errors by terrain.

## 5. Web GUI: two tabs

**Tab 1 — Bag + DBM calibration**

- Load a ROS2 bag, choose topics/terrain/time range, and inspect trajectory, velocity, steering, and yaw rate with synchronized playback.
- Overlay DBM predictions from the same initial state and recorded commands; adjust physical parameters, actuator response, and command alignment interactively.
- Fit parameters on a selected calibration segment, inspect errors on a separate segment, and save/load terrain presets.
- Export real bins and launch synthetic generation from the saved preset; show progress and browse `syn_data → terrains → sim / real`.

**Tab 2 — Bag + trained transformer**

- Load a ROS2 bag and either architecture’s `.ckpt`; select history, start time, and prediction horizon.
- Overlay measured and predicted trajectories using recorded future commands, with playback and position/heading errors.
- Use checkpoint preprocessing and normalization; show fixed-window predictions without injecting future ground truth.

## 6. Implementation order and completion checks

1. Torch DBM/environment + math utilities; compare against the existing Torch reference, including actuator state.
2. Shared binary schema + both generators; verify FKD loading, command alignment, bounded memory, and session-disjoint splits.
3. DBM calibration GUI + terrain preset export + synthetic-generation controls.
4. Both transformers + generalist/specialist training; smoke-test training and checkpoint reload for each architecture on a separate tiny fixture.
5. Transformer GUI; verify predictions match CLI evaluation on the same bag window.
6. Add concise setup instructions and commands for conversion, generation, both training stages, evaluation, and GUI launch.

Deliverable: a repeatable bag-to-dataset-to-checkpoint workflow, with a browsable calibration/evaluation GUI. Full generalist training begins once its dataset is supplied.
