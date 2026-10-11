"""Start the OptCar web GUI with ``python3 web_gui.py``."""
import sys

from optcar.cli import main


if __name__ == "__main__":
    main(["gui", *sys.argv[1:]])
