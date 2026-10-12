# Dataset format

```text
datasets/
  generalist_dataset/                 # Initially empty
  syn_data/terrains/<terrain>/
    dbm_params.yaml
    real/<run>/                       # ROS2 recordings
    sim/<run>/                        # DBM episodes
```

Synthetic generation writes raw bins and metadata. Dataset builder writes processed bins under `<chosen folder>/processed/{sim|real}/{terrain}/{train|test|valid}/`. Each source/terrain manifest records physical shapes and recording splits. Earlier combined generation/writer outputs can also contain `processed/data/*.bin` and remain readable.

Processed bins are float32 `[windows, history_steps + future_steps, 42]`:

| Columns | Contents |
|---|---|
| 0–12 | World pose and linear/angular velocity |
| 13–25 | State in the current body's frame |
| 26–38 | Previous-frame pose increments and body velocities |
| 39–40 | Commanded velocity (m/s), steering (rad) |
| 41 | Vehicle ID |

Quaternions use `w,x,y,z`. Raw bins contain world state plus commands (15 features). Imported bags retain commands at their recorded samples; synthetic bins use WheeledLab's post-step state/command logging. These raw stages retain the final recorded command. For Twist command topics, `angular.z` means steering angle. The optional vehicle ID is outside the 41 state/action channels.

Dataset builder's `future_steps=50` counts transitions, so its saved metadata has `future_horizon=51` and 300 physical rows with 250 history samples. The reference **model** config uses `future_horizon_steps=50`, including the current sample: 299 logical rows, history 0–249 and future 249–298. The training loader reads each bin using its physical metadata, then takes the first 299 rows. Both existing 300-row OptCar bins and original 299-row WheeledLab bins work with the same reference H250/F50 model; no rebuilding or reshaping files is required. The first future delta has zero position and identity quaternion.

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

Match these physical dimensions and frame conventions to your files. Set `session_disjoint_splits` only after splitting by complete source sessions. Training uses a separate `dataloader_config()` factory with dataset/test paths and per-folder file/window limits. Recorded parent folders resolve to their train/test files; the copied WheeledLab training entry applies its seeded 90%/10% window split to the selected main data. The reference CORL26 model leaves non-quaternion values unchanged and unit-normalizes quaternion groups.
