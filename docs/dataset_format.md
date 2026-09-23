# Dataset format

```text
datasets/
  generalist_dataset/                 # Initially empty
  syn_data/terrains/<terrain>/
    dbm_params.yaml
    real/<run>/                       # ROS2 recordings
    sim/<run>/                        # DBM episodes
```

Each generated run contains `manifest.json`, `raw/*.bin`, and `processed/data/*.bin`. The manifest records episode splits, shapes, parameters, and seeds. Keep complete bags/episodes in one split. Repeat generation with the same settings to resume; use `--run <name>` when changing settings.

Processed bins are float32 `[windows, history_steps + future_steps, 42]`:

| Columns | Contents |
|---|---|
| 0–12 | World pose and linear/angular velocity |
| 13–25 | State in the current body's frame |
| 26–38 | Previous-frame pose increments and body velocities |
| 39–40 | Commanded velocity (m/s), steering (rad) |
| 41 | Vehicle ID |

Quaternions use `w,x,y,z`. Command `t` advances state `t` to `t+1`. Raw bins contain world state plus commands (15 features); the final command is a zero placeholder. For Twist command topics, `angular.z` means steering angle.

OptCar's `future_steps=50` counts transitions. The equivalent WheeledLab FKD `future_horizon` is **51**, because it includes the current state. With 250 history steps, a window therefore has 300 rows. Existing 299-row FKD windows require `future_steps=49` in both data and model configs.

## Existing generalist bins

Either place OptCar datasets with manifests under `generalist_dataset/`, or provide `train/*.bin`, `val/*.bin`, and `test/*.bin` with this `schema.json`:

```json
{
  "features": 42,
  "dtype": "float32",
  "history_steps": 250,
  "future_steps": 50,
  "dt": 0.02,
  "session_disjoint_splits": true
}
```

Match these dimensions and frame conventions to your files. Set `session_disjoint_splits` only after splitting by complete source sessions. Normalization uses training data only and is retained during specialist training.
