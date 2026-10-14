#!/usr/bin/env bash
# Run a command with MuJoCo's OSMesa software renderer, for machines without a display or EGL.
#   OSMESA_PREFIX=/path/to/env-with-mesalib scripts/headless.sh uv run python -m ur_pusht.render ...
# The prefix must contain lib/libOSMesa.so.8 (e.g. `conda create -p <prefix> -c conda-forge mesalib=24.3`).
set -euo pipefail
: "${OSMESA_PREFIX:?set OSMESA_PREFIX to an environment that provides lib/libOSMesa.so.8}"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$OSMESA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$OSMESA_PREFIX/lib/libOSMesa.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$@"
