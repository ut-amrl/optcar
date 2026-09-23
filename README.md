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

From the WheeledLab repository root:

```bash
cd source/optcar
pip install -e '.[ros]'
optcar init
optcar gui
```

Open **http://127.0.0.1:5050**. Run the remaining commands from `source/optcar`.

- **DBM tab:** load a bag, tune/fit parameters, save a terrain preset, and export real or synthetic data.
- **Transformer tab:** load a bag and `.ckpt` to compare predicted and recorded trajectories.

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
