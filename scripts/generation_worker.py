"""Internal tmux worker for OptCar synthetic collections."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcar.datagen.generation_jobs import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker(sys.argv[1]))
