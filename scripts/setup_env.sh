#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash scripts/setup_env.sh' >&2
  exit 2
fi
case "${1:-}" in
  '') ;;
  -h|--help)
    echo 'Usage: bash scripts/setup_env.sh'
    echo 'Install GPU-enabled requirements.txt and verify every visible NVIDIA GPU.'
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if ! command -v conda >/dev/null 2>&1; then
  echo 'Conda is not available. Open a shell with Conda initialized, then rerun setup.' >&2
  exit 1
fi
if ! conda run -n optcar python --version >/dev/null 2>&1; then
  conda env create -f environment.yml
fi
if ! conda run -n optcar python -c 'import shutil; assert shutil.which("tmux")' >/dev/null 2>&1; then
  conda install -n optcar -c conda-forge tmux -y
fi
echo 'Installing GPU-enabled requirements in the optcar environment.'
# The CUDA local-version pin replaces an existing CPU-only Torch build.
conda run --no-capture-output -n optcar python -m pip install -r requirements.txt -e '.[bags]'
conda run --no-capture-output -n optcar python -m pip check
conda run --no-capture-output -n optcar python -c '
import importlib
import shutil
import subprocess
import torch
from optcar.utils.devices import available_devices
for module in ("numpy", "scipy", "yaml", "flask", "rosbags", "rich", "joblib", "matplotlib",
               "optcar.gui.server", "optcar.datagen.raw_bags", "optcar.datagen.build_dataset",
               "optcar.datagen.generation_jobs", "optcar.train", "optcar.evaluate"):
    importlib.import_module(module)
print("Verified runtime imports for conversion, generation, dataset building, plotting and training.")
tmux = shutil.which("tmux")
assert tmux, "tmux is missing"
print(subprocess.check_output([tmux, "-V"], text=True).strip())
print(f"Torch {torch.__version__}; CUDA build: {torch.version.cuda}")
devices = available_devices()["devices"][1:]
print(f"Detected {len(devices)} GPU(s)")
for gpu in devices:
    device = gpu["id"]
    x = torch.ones((2, 2), device=device)
    assert (x @ x).sum().item() == 8
    torch.cuda.synchronize(device)
    print("Verified {}: {} ({} GiB)".format(device, gpu["name"], gpu["memory_gib"]))
if not devices:
    print("No NVIDIA GPU is available to Torch; the GUI will offer CPU. Check driver and GPU access if a GPU was expected.")
'
echo 'Ready: conda activate optcar && python3 web_gui.py'
echo 'Restart any running OptCar GUI after updating Torch.'
