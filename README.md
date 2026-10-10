# OptCar

Torch DBM data generation, ROS2 bag conversion, and two-stage transformer training. [Paper](https://arxiv.org/abs/2607.13319).

<p align="center">
  <img src="assets/images/optcar_title.png" alt="OptCar overview across mixed terrain" width="100%" />
</p>

<p align="center">
  <img src="assets/images/optcar_method.png" alt="OptCar model architecture and fine-tuning recipe" width="100%" />
</p>

**Code in progress.**

## Install and open the GUI

From this OptCar folder (all converter code and scripts live here):

```bash
bash scripts/setup_env.sh
conda activate optcar
optcar gui --bag-root /home/rwik/research/jumpracing/rosbags/corl26
```

Open **http://127.0.0.1:5050**. `setup_env.sh` creates the `optcar` Conda environment with Python 3.11, installs CPU Torch for the existing workbench, then installs `requirements.txt` and OptCar. CUDA/training setup can follow later. `pyproject.toml` already describes package dependencies; `requirements.txt` and `environment.yml` make this setup explicit.

- **ROS2 bags → bins:** scan a folder recursively, inspect duration and topics, choose bags, and select an output folder. Conversion runs in the background with overall and current-bag progress, elapsed time, an estimated remaining time, results, and cancellation. Completed bins are retained on cancellation; refreshing the page reconnects to an active job while the server remains running.
- **Raw bin viewer:** open a dataset folder, nested bin folder, or single `.bin`. Expand the folder browser and click a file. Inspect world XY motion with zoom/pan, play or scrub through every sample, and view position, wxyz orientation, world linear/angular velocity, and commands with exact cursor values. Beside the world trajectory, two plots show commanded velocity versus measured forward body velocity, and the steering command. Use **Action plot time range → Around current** to set seconds before and after the selected sample; both action plots follow playback/scrubbing with the current sample at `0 s`. **Full recording** restores the complete time axis. Expand **Y-axis limits** to set independent velocity and steering minimum/maximum values; leave either limit blank for automatic scaling. Manual limits stay fixed while scrubbing or playing. These controls use the viewer timestep, and narrow windows read their samples directly rather than relying on the sampled overview. Forward velocity uses the auxiliary body twist when present, otherwise a full-quaternion rotation of the main bin’s world velocity. After conversion, **View raw bins** opens this tab with the output folder filled in; the folder remains editable.
- **DBM calibration:** the manual web estimator copied from WheeledLab. Defaults use its CORL26 AlphaTruck preset (`len_f = len_r = 0.245 m`), shared with the backend and `configs/dbm.yaml`. The left folder explorer shares the raw viewer's expandable tree, filtering, selection, and duration/sample metadata. Click a raw 15/17-column bin, scrub the current sample, set a prediction horizon and command delay, edit all 13 DBM parameters, and use **Set & Rollout** to overlay the predicted future in the current sample’s body frame. **Tune DBM** in the raw viewer opens that exact bin and sample. World-trajectory segment selection and YAML saving/loading, plus circle/figure-eight/Dubins/custom reference trajectories, are preserved. Aggregate comparisons are not exposed in the interface. Terrain presets use OptCar’s existing preset storage. **Bag fitting & dataset tools** opens the separate, existing ROS2-bag workflow: bounded least squares fits selected parameters to measured one-step transitions only after **Fit DBM parameters** is clicked; export and synthetic dataset generation are available there too. **Back to manual bin tuning** returns to the manual tool.
- **Transformer evaluation:** existing checkpoint and trajectory comparison.

The raw-bin reader is copied and adapted from WheeledLab's `dbm_estimator/extractor.py`; plot grouping and multi-environment ordering follow `plot_sim_bins.py`. It reads 15/17-column float32 bins, detects timestep and environment count from metadata, and supports manual layout overrides for standalone files. Dataset config files are read as text without executing Python. Large files use memory mapping and a sampled overview; the current sample is read exactly, including the final row. Signal ranges cover the full recording, and nonfinite values appear as gaps with a warning. This viewer expects raw state/action bins, before training-window processing. The manual DBM tab uses these same raw bins. Its plots and handlers are copied from `wheeledlab_utils/dbm_estimator`, with reference generators copied from `wheeledlab_tasks/mppi`; all adapted code and the locally served Plotly asset are inside OptCar. A horizon of H steps shows the current state plus H future states, clipped at the recording end. Positive command delay uses earlier recorded commands; rollouts report an error when the delay exceeds available command coverage or the parameters produce nonfinite results.

## Raw ROS2 bags to bins

This first stage copies WheeledLab's raw converter into `optcar/datagen/wheeledlab_rosbag_converter.py`. Its `_align_asof`, quaternion normalization, `process`, `_stack_rows`, and raw dataset configuration conventions are retained. The reader is adapted to `rosbags`, with progress and cancellation callbacks; no ROS Python installation or WheeledLab imports are required. `raw_bags.py` handles discovery, selection, metadata, output paths and jobs. The GUI and terminal both call `convert_raw_bags`, which invokes the copied aligner and row stacker.

```bash
optcar scan-bags /home/rwik/research/jumpracing/rosbags/corl26
# Full metadata, including per-topic counts:
optcar scan-bags /home/rwik/research/jumpracing/rosbags/corl26 --json

# Convert selected recordings with an in-place terminal progress display:
bash scripts/bags_to_bins.sh /home/rwik/research/jumpracing/rosbags/corl26 \
  --select data-collect/road/v2.0-s0.0 data-collect/grass/v2.0-s0.0

# Omit --select to take every bag; --exclude accepts relative paths to omit.
# --output /path/to/bins selects a destination; otherwise ROOT's parent/bins.
# --config configs/raw_bags.yaml loads the same settings exposed in the GUI.
```

For the `corl26` root, the default output is:

```text
rosbags/bins/
  data/corl26/data-collect/road/v2.0-s0.0.bin
  data/corl26/data-collect/grass/v2.0-s0.0.bin
  metadata/
    dataset_config.py      # Compatible with WheeledLab's raw processor
    schema.json            # Dtype, dimensions, frame and column names
    conversion.json        # Source bag, settings, shape and topic metadata per bin
    last_job.json          # Complete, failed and cancelled results
```

Default raw bins are **float32 [T, 15]**: position (3), quaternion wxyz (4), world linear/angular velocity (6), commanded velocity and steering (2). Every row retains its recorded command. The optional 17-column mode adds the unshifted command pair, matching WheeledLab's `include_cmd_vel` layout. This raw stage does **not** generate training windows or assign splits; the existing `optcar convert` command below does that separately.

New conversions also read `/super_odometry/twist` and save its six linear/angular velocity channels in `aux/<same relative path>.bin`. This preserves the recorded body twist separately from the main bin’s world twist. `conversion.json` links each companion and records frame IDs, shape, timestep, start time, and missing values. Auxiliary samples use the same bag timestamps and sample-and-hold grid as the main bin, without shifting commands or trimming the main recording. Before/after auxiliary coverage, missing values are stored as NaN and displayed as gaps. Missing auxiliary topics do not block conversion. The viewer only lists main `data/` bins; training inputs remain 15/17 columns. Existing bins work with derived body velocity; select overwrite on the bags you want to reconvert with auxiliary data. The GUI’s **Auxiliary data** setting and CLI `--aux-twist-topic` choose another body twist topic; an empty name disables auxiliary output.

Defaults match WheeledLab: `/super_odometry/pose`, `/super_odometry/twist_wf` in the world frame, `/cmd_vel`, bag timestamps, 0.02 s timestep, zero action shift, and sample-and-hold alignment over the shared topic time range. Positive action shift applies commands later. Body-frame velocities can optionally be rotated into world coordinates. Maximum topic gaps are recorded in the metadata; raw alignment keeps WheeledLab's forward-fill behavior across gaps.

Existing bins are protected unless `--overwrite` (or its GUI checkbox) is selected. Overwrite replaces only selected outputs. Outputs are written atomically, and simultaneous conversions to the same destination are rejected. Different settings require a new destination so sampling rates and schemas cannot be mixed. Missing/malformed metadata and missing topics are shown per bag; source metadata is never repaired automatically. Missing metadata can be regenerated with `ros2 bag reindex /path/to/bag` in a suitable ROS environment.

Run the integration checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_raw_bags.py tests/test_raw_bins.py
```

The flag avoids unrelated ROS pytest plugins when your shell has sourced ROS. Install test dependencies with `pip install -e '.[dev]'`.

## Prepare data from the terminal

Set your bag paths and edit the topics in `configs/rosbag.yaml`:

```bash
TRAIN_BAG=/data/road_train
VAL_BAG=/data/road_validation
TEST_BAG=/data/road_test

optcar topics "$TRAIN_BAG"
optcar convert "$TRAIN_BAG" --terrain road --split train
optcar convert "$VAL_BAG" --terrain road --split val
optcar convert "$TEST_BAG" --terrain road --split test
optcar fit "$TRAIN_BAG" --terrain road
optcar generate --terrain road --episodes 10000 --device cuda
```

Repeat for other terrains. Use separate bags for each split. `configs/datagen.yaml` controls Bézier velocity, random-walk steering, noise, and episode length. Replace `cuda` with `cpu` if needed.

## Train

Supply generalist data in `datasets/generalist_dataset/` first; it is initially empty. See [dataset format](docs/dataset_format.md) for existing FKD bins.

```bash
# Generalist → specialist (real + synthetic data)
optcar train --config configs/train_generalist.yaml --device cuda
optcar train --config configs/train_specialist.yaml --device cuda
```

Final checkpoint: `checkpoints/film_specialist/specialist_best.ckpt`.

For cross-attention, use a separate checkpoint pair:

```bash
optcar train --config configs/train_generalist.yaml --architecture cross_attention \
  --output checkpoints/cross_generalist --device cuda
optcar train --config configs/train_specialist.yaml --architecture cross_attention \
  --checkpoint checkpoints/cross_generalist/generalist_best.ckpt \
  --output checkpoints/cross_specialist --device cuda
```

## Evaluate

```bash
optcar evaluate --checkpoint checkpoints/film_specialist/specialist_best.ckpt \
  --datasets datasets/syn_data/terrains --real-only
```

Use `optcar <command> --help` for options, including training resume and prediction export.

## Code

- `optcar/models/`: `dynamic_bicycle.py`, `dbm_calibration.py`, `fkd_transformer.py`.
- `optcar/datagen/`: `dbm_to_dataset.py`, `rosbag_to_dataset.py`.
- `optcar/dataset.py`: binary format, writer, loader, and balanced sampling.
- `optcar/train.py`, `optcar/evaluate.py`: training and evaluation.
- `optcar/gui/`, `optcar/utils/`: web interface and shared helpers.
