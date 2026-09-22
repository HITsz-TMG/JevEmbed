#!/usr/bin/env bash
set -euo pipefail
JEVEMBED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$JEVEMBED_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
JEVEMBED_PYTHON="${JEVEMBED_PYTHON:-python3}"
cd "$JEVEMBED_ROOT"
exec "$JEVEMBED_PYTHON" -m jevembed "$@"
