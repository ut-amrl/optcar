#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if ! conda run -n optcar python --version >/dev/null 2>&1; then
  conda env create -f environment.yml
fi
conda run -n optcar python -m pip install 'torch>=2.1,<3' --index-url https://download.pytorch.org/whl/cpu
conda run -n optcar python -m pip install -r requirements.txt -e '.[bags]'
echo 'Ready: conda activate optcar && optcar gui'
