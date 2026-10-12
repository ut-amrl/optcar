# OptCar

Prepare vehicle datasets, calibrate a dynamic bicycle model (DBM), and train FKD delta-prediction transformers.

**Rwik Rana** · [rwikrana@gmail.com](mailto:rwikrana@gmail.com)

[![Paper](https://img.shields.io/badge/Paper-333333?style=for-the-badge)](https://arxiv.org/abs/2607.13319)
[![Website](https://img.shields.io/badge/Website-333333?style=for-the-badge)](https://amrl.cs.utexas.edu/optcar/)

<p align="center"><img src="assets/images/optcar_title.png" alt="OptCar across mixed terrain" width="100%" /></p>
<p align="center"><img src="assets/images/optcar_method.png" alt="Model architecture and fine-tuning" width="100%" /></p>

## Quick start

### 1. Install and open the GUI

Requires Conda. Skip cloning if you already have the repository.

```bash
git clone git@github.com:ut-amrl/optcar.git
cd optcar
bash scripts/setup_env.sh
conda activate optcar
python3 optcar_gui.py
```

Open **http://127.0.0.1:5050**. Setup installs dependencies, tmux and CUDA-enabled PyTorch, then checks the available NVIDIA GPUs.

### 2. Prepare the data

Follow these GUI tabs; skip conversion or generation if you already have raw bins.

1. **ROS2 bags → bins:** scan your bag folder, select recordings, check topics and timestep (`0.02 s` by default), choose an output folder and convert. Inspect results in **Raw bin viewer**.
2. **DBM calibration:** select a raw bin, adjust parameters, then **Set & Rollout**. Save the YAML using your chosen folder and filename. Skip this if you already have a calibrated config.
3. **Synthetic raw bins:** enter the DBM config path, run name, GPU, batch size and run count. Set command ranges and optional domain randomization, then generate. Results go under `<DBM config folder>/<run name>/`. Inspect them in **Visualize synthetic data**, choosing a batch index.
4. **Dataset builder:** add real and/or synthetic raw folders, select bins and splits, and choose the output folder and device. For the supplied model, use **250 history samples, 50 future transitions and dt 0.02 s**. Set the stride; the default output size is **10,000 windows per bin**. Build the dataset.
5. **Dataset viewer:** add processed split folders, set each folder's file/window limits (`-1` means all), and **Load dataset**. Inspect indices or shuffle. Enter a full config path, e.g. `<repo>/configs/corl26_data.py`, and **Save dconfig.py**. This creates the dataset config used below.

Generation and build dashboards show progress and tmux attach commands. Those jobs continue after the GUI closes.

### 3. Train

In another terminal, from the repository root:

```bash
conda activate optcar
MPLBACKEND=Agg python3 optcar/train.py \
  --dataset-config configs/corl26_data.py \
  --model-config configs/corl26_model.py \
  --learning-config configs/corl26_learning.py \
  --device cuda:0 \
  --output checkpoints/corl26_delta
```

Use the dataset config saved above and a fresh output folder. Add `--check` to validate without training. The supplied model has **321,219 trainable parameters**.

### 4. Monitor and inspect results

```bash
tensorboard --logdir checkpoints/corl26_delta/lightning_logs
```

Open **http://127.0.0.1:6006** for metrics and validation plots. Checkpoints and copies of all three configs are saved in `checkpoints/corl26_delta/`. For WandB, run `wandb login` and add `--wandb` to the training command.

## Configuration and data details

<details>
<summary>Configs, training and fine-tuning</summary>

- **Dataset:** your exported `configs/corl26_data.py` defines folders, `num_runs` (bin files), `num_trajectories` (windows), selection seed and loader settings. A machine-specific dataset mix is not bundled.
- **Model:** [`configs/corl26_model.py`](configs/corl26_model.py) defines the CTX/FiLM delta model: embedding 64, 4 attention heads, 2 context layers and 2 decoder layers.
- **Learning:** [`configs/corl26_learning.py`](configs/corl26_learning.py) sets 500 epochs, batch size 4096, AdamW at `1e-3`, weight decay `0.01`, 1% warmup and cosine scheduling. Loss weights: absolute position `0.005`, delta position `1.0`, delta-quaternion geodesic `0.3`.

To fine-tune, add `--checkpoint /path/to/model.ckpt` to the training command; this initializes weights with a fresh optimizer. To resume an interrupted run:

```bash
python3 optcar/train.py --resume checkpoints/corl26_delta/last.ckpt --device cuda:0
```

Resume uses the saved configs and existing output folder. Reapply original CLI overrides for device, seed, batch size or epochs. Effective settings are recorded in `run_config.json`.

Main dataset folders are split 90%/10% for training/validation. Separate test folders are evaluated every epoch; `--no-test` skips that loader. Training/validation default to 4 workers each (`--train-num-workers`, `--val-num-workers`), separate from the builder's 16 file-loading threads.

The model's future horizon includes the current sample: it uses 299 rows of a 300-row window, with history 0–249 and future 249–298. At dt 0.02 s, that is 49 future transitions (0.98 s).

**Current limitations:** `val/learn/loss_epoch` reports zero, so the default `best` checkpoint does not reliably rank validation quality; inspect component metrics and epoch/last checkpoints. The GUI's **Transformer evaluation** tab expects a separate checkpoint format and does not load these Lightning training checkpoints. Use training validation plots for this workflow.

</details>

<details>
<summary>Bin formats, frames and output folders</summary>

- **Raw bins:** float32 `[batch, steps, 15]` — position (3), quaternion **wxyz** (4), world linear/angular velocity (6), velocity/steering commands (2). Optional 17-column bins also retain unshifted commands. Body twist is stored in an auxiliary bin when available.
- **Processed bins:** `[windows, history + future transitions, 42]` — world state (13), body state (13), delta state (13), commands (2), vehicle/file ID (1). The settings above produce up to `[10000, 300, 42]` per bin. Preview counts are windows, not timesteps.
- **Body frame:** anchored at the final history sample's position and full quaternion. Delta position/orientation uses the previous sample's frame; delta velocities use each sample's own body frame. Quaternion sign continuity is preserved.

```text
<output>/processed/<sim or real>/<terrain>/<train, valid or test>/*.bin
```

An explicitly selected, separate `processed` directory is used directly. Windows never cross trajectories; recordings stay together during builder splits. Windows accumulate across input files; only the last bin per source/terrain/split may be smaller.

File limits apply before window limits. A selection seed chooses reproducible subsets; otherwise selection follows sorted order. Viewer shuffle only changes display order. Training horizons come from the model config; viewer horizons are saved separately in the dataset config.

</details>

<details>
<summary>Conversion, generation and devices</summary>

Bag conversion defaults to `/super_odometry/pose`, `/super_odometry/twist_wf` and `/cmd_vel`, using bag timestamps and last-observation alignment at dt 0.02 s. Positive action shift delays commands. Output preserves input structure under `output/data/`; the default output is the bag folder's parent plus `bins/`.

The terminal uses the same converter:

```bash
bash scripts/bags_to_bins.sh /path/to/bags \
  --output /path/to/bins --config configs/raw_bags.yaml
```

Synthetic generation saves one batched bin per run. Batch size 32, duration 10 s and dt 0.02 s gives 32 trajectories of 500 samples (960,000 bytes). States are logged after applying each row's command.

Domain randomization supports uniform and normal ranges. Vehicle parameters are shared within a batch; **Resample vehicle parameters each run** draws a new set per run. Trajectory command biases are sampled independently. The base seed plus run index controls reproducibility; **Fixed control seed** repeats commands. Disable Bézier stretching if commands should not always reach both bounds.

Generation/build jobs each use one selected device; multiple jobs can use different GPUs. GPU indices follow PyTorch's visible ordering. Bag decoding uses CPU. The single `requirements.txt` installs Torch 2.11.0 with CUDA 12.8 and supports CPU execution. Rerun setup to update dependencies, then restart the GUI.

Use `python3 optcar_gui.py --help` and `python3 optcar/train.py --help` for more options. Third-party attribution is in [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).

</details>

## Citation

```bibtex
@article{rana2026adapting,
  title   = {Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains},
  author  = {Rana, Rwik and Quattrociocchi, Jesse and Ellis, Christian and Tsoi, Nathan and Warnell, Garrett and Biswas, Joydeep},
  journal = {Proceedings of The 10th Conference on Robot Learning},
  year    = {2026}
}
```
