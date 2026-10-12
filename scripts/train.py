#!/usr/bin/env python3
"""Run the self-contained WheeledLab FKD training entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optcar.fkd_train import main


if __name__ == "__main__":
    main()
