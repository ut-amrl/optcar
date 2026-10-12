# OptCar

Vehicle dynamics data preparation, DBM calibration and FKD transformer training.

Author: **Rwik Rana** · [rwikrana@gmail.com](mailto:rwikrana@gmail.com)

[Paper](https://arxiv.org/abs/2607.13319)

<p align="center"><img src="assets/images/optcar_title.png" alt="OptCar across mixed terrain" width="100%" /></p>
<p align="center"><img src="assets/images/optcar_method.png" alt="Model architecture and fine-tuning" width="100%" /></p>

## Setup

From the repository root:

```bash
bash scripts/setup_env.sh
conda activate optcar
python3 optcar_gui.py
```

Open http://127.0.0.1:5050. The setup script creates a Python 3.11 environment,
installs `requirements.txt` and the editable package, checks imports and tmux,
and verifies every visible NVIDIA GPU. The single requirements file installs
Torch 2.11.0 with CUDA 12.8; CPU execution is also available. To update an existing
environment, rerun the setup script or install manually:

```bash
python3 -m pip install -r requirements.txt -e '.[bags]'
```

GUI options include `--bag-root`, `--dataset-root`, `--host` and `--port`.
Use `python3 optcar_gui.py --help` for details. Restart the GUI after updating
packages or changing `CUDA_VISIBLE_DEVICES`.

## Workflow

1. **ROS2 bags → bins:** scan a folder, select bags, choose the output folder and
   convert. Duration and topic names appear on each card. Default timestep is
   0.02 s. The GUI and `scripts/bags_to_bins.sh` call the same converter.
2. **Raw bin viewer:** browse folders and select a bin; play or scrub samples.
   View world trajectory, commanded velocity, forward body velocity and steering.
   Set action time ranges around the current sample and optional Y-axis limits.
3. **DBM calibration:** select a raw bin, edit parameters and click **Set & Rollout**.
   Defaults use the AlphaTruck (`len_f = len_r = 0.245 m`). Set both config folder
   and filename before saving. Segment selection, saved segments, command delay,
   and circle/figure-eight/Dubins/custom references are available. The separate
   automatic bag fitter uses bounded least squares on selected parameters.
4. **Synthetic raw bins:** load your DBM YAML, set a run name, timestep, duration,
   batch size, commands and optional domain randomization. Output goes to
   `<config folder>/<run name>/`. Each run saves one `[batch, steps, 15]` bin.
   Each collection has its own tmux session, shown in the run dashboard.
5. **Visualize synthetic data:** the raw viewer with a batch-index selector.
   For a batch of 32, indices 0–31 select trajectories inside each bin.
6. **Dataset builder:** select raw sources and recording splits, set history,
   future transitions, stride, processing device, file workers and windows per
   output bin. Jobs run in tmux and appear in the build dashboard.
7. **Dataset viewer:** browse processed folders, copy/add their paths, and set
   per-folder `num_runs` (files) and `num_trajectories` (windows) limits. `-1` selects
   all. Load the dataset, inspect an index or shuffle the viewing order. Set a
   full `.py` path to save a dataset config; loading that config fills the fields.
8. **Transformer evaluation:** compare predictions from evaluation checkpoints
   with recorded trajectories.

Generation and building each use one selected GPU. Multiple jobs can use the
same or different GPUs; a single job is not distributed across devices. Device
indices follow Torch's visible ordering. Bag decoding runs on the CPU.

## Raw bins and collection

The converter uses bag timestamps and last-observation alignment over the shared
range of the selected topics. Defaults are `/super_odometry/pose`,
`/super_odometry/twist_wf`, `/cmd_vel`, dt 0.02 s and zero action shift. Positive
action shift applies commands later. Optional body-frame input velocities are
rotated to world coordinates. Conversion preserves input folder structure under
`output/data/` and writes metadata beside it. The default output is the input
folder's parent plus `bins/`.

Raw float32 rows have 15 columns: position (3), quaternion **wxyz** (4), world
linear/angular velocity (6), velocity/steering commands (2). Optional 17-column
bins add the unshifted command pair. Auxiliary body twist is stored separately;
the viewer uses it for forward velocity when available, otherwise rotates world
velocity using the recorded quaternion. Metadata supplies timestep and batch
count; standalone bins accept explicit layout settings.

Synthetic rows log the state **after** applying that row's command. A batch of
32, duration 10 s and dt 0.02 saves `32 × 500 × 15 × 4 = 960,000` bytes per run.
Ten runs save ten bins containing 320 trajectories.

Domain randomization accepts uniform `[min, max, uniform]` and normal
`[mean, std, normal]` ranges for selected physical/control parameters. Parameters
are shared within a batch; **Resample vehicle parameters each run** draws a new
set per run. Per-trajectory velocity/steering biases are sampled independently.
The base seed plus run index controls reproducibility; **Fixed control seed**
repeats the same control sequence. The default Bézier stretching reaches both
command bounds. Disable it to keep the sampled control points' original range.

## Build and load datasets

The builder loads input bins with 16 file threads by default. It processes all
valid trajectories in each bin on the selected device, creates sliding windows,
and transforms them into world, current-body and delta states. Invalid whole
trajectories are excluded. Each window stays inside one trajectory; splits keep
recordings together.

The output layout is:

```text
<output>/processed/<sim or real>/<terrain>/<train, valid or test>/*.bin
```

If the selected output already names a separate `processed` directory, it is
used directly. **Windows per output bin** defaults to 10,000. Windows carry
across input files until a bin is full; only the last bin per source/terrain/split
may be smaller. With history 250 and future 50 transitions, full output bins are
`[10000, 300, 42]`: world (13), body (13), delta (13), commands (2), vehicle/file
ID (1). Preview counts represent windows, not timesteps.

The current sample is `history_steps - 1`. Body states use its full quaternion
and position. Delta position/orientation uses the previous sample's frame;
delta velocities are absolute velocities in each sample's own body frame.
Quaternion sign continuity is maintained without forcing positive w.

The dataset config specifies folders, file/window limits, dtype/device and loader
options. File selection precedes trajectory limits. Without a selection seed,
sorted files/windows are selected; a seed uses deterministic random subsets.
The viewer's shuffle seed changes display order only. The config contains a
`dataloader_config()` factory; viewer horizons are saved separately as a
`viewer_data` literal because training reads horizons from the model config.
GUI config loading parses literals without executing Python.

## Train

There is one training implementation and Python entrypoint: `optcar/train.py`.
The `optcar train` command calls it as well. Configs are separate because they
control different parts of the run:

- `configs/corl26_model.py`: CTX/FiLM delta model, **321,219 trainable parameters**,
  embedding size 64, 4 attention heads, 2 context layers and 2 decoder layers.
- `configs/corl26_learning.py`: 500 epochs, batch size 4096, AdamW learning rate
  0.001, weight decay 0.01, 1% warmup and cosine scheduling. Loss weights are
  0.005 absolute-position MSE, 1.0 delta-position MSE and 0.3 delta-quaternion
  geodesic loss. Gradient clipping is disabled.
- Your dataset factory, exported from Dataset viewer, e.g. `configs/corl26_data.py`.
  No machine-specific dataset mix is bundled.

Run from the repository root in the installed environment:

```bash
python3 optcar/train.py \
  --dataset-config configs/corl26_data.py \
  --model-config configs/corl26_model.py \
  --learning-config configs/corl26_learning.py \
  --device cuda:0 \
  --output checkpoints/corl26_delta
```

Add `--check` to validate configs, selected data and model without training or
writing outputs. Add `--checkpoint /path/to/initial.ckpt` to initialize weights
for a new run. `--resume /path/to/run/last.ckpt` continues a run using its saved
config factories. Reapply any original CLI device/seed/batch/epoch overrides;
effective settings are recorded in `run_config.json`. Resume writes into the
checkpoint's existing directory. Training config factories execute as Python,
so use config files you trust.

Training history has 250 samples; future has 50 **including the current sample**.
The loader uses 299 rows from a 300-row processed window: history 0–249, future
249–298. The first future delta has zero position and identity orientation. The
future spans 49 transitions (0.98 s at dt 0.02). Existing bins need no rebuilding.

Main dataset folders are split 90%/10% into training and validation. Test folders
are a separate validation loader evaluated each epoch; `--no-test` skips it.
Default workers are 4 for training and 4 for validation, configurable with
`--train-num-workers` and `--val-num-workers`. These are separate from builder
workers and output-bin size. The entrypoint raises its open-file limit toward
65,536 before loading bins.

Lightning saves config factories, `initial.ckpt`, per-epoch checkpoints,
`last.ckpt` and local TensorBoard logs. Add `--wandb --wandb-project-name optcar`
after `wandb login` for WandB. Validation plots include positions, quaternions,
commands and trajectory overlays; the dataset config controls plotting frequency.
For headless servers, set `MPLBACKEND=Agg`.

The current loss implementation reports zero for `val/learn/loss_epoch`; this
cleanup preserves that numerical behavior. The default `best` checkpoint does
not reliably rank validation quality. Review component metrics and epoch/last
checkpoints instead.

## Repository

- `optcar/datagen/`: bag conversion, synthetic collection and dataset building.
  `collection/` groups controllers, parameter definitions, randomization and saving.
- `optcar/dataset.py`: binary windows, frames, manifests and file selection.
- `optcar/fkd_dataloader.py`: training tensor slicing and loading.
- `optcar/models/`: bicycle dynamics, calibration and FKD architectures.
- `optcar/utils/math.py`: quaternion operations, continuity and frame transforms.
- `optcar/gui/`: the local web interface and DBM tool.

Run tests with `python3 -m pytest -q`. Numerical regression fixtures are frozen
independent outputs for frames, dynamics, controllers, loading and model
forward/loss/gradients/scheduling. They require no external source checkout.
Third-party attribution is retained in `THIRD_PARTY_NOTICES`.
