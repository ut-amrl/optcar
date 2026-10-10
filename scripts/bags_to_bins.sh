#!/usr/bin/env bash
set -euo pipefail
# With optcar activated, identical backend and options to the web workbench.
exec python -m optcar bags-to-bins "$@"
