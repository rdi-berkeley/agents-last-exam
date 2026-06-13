#!/usr/bin/env bash
# Verify — celegans: RTENV builds via uv + the imaging/Qt stack imports headless
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/psychology_neuro/celegans_neuron_tracking/137}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -f "$RTE/pyproject.toml" || fail "runtime_env/pyproject.toml missing"
export UV_PROJECT_ENVIRONMENT="$CANON/output/.verify_venv" UV_CACHE_DIR="$CANON/output/.uv_cache"
export QT_QPA_PLATFORM=offscreen
S="uv sync --frozen --project $RTE"; [ -f "$RTE/uv.lock" ] || S="uv sync --project $RTE"
echo "[verify] $S ..."; $S >/dev/null 2>&1 || $S || fail "uv sync failed (missing system lib?)"
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY' || fail "imaging/Qt imports failed"
import cv2, h5py, numpy, scipy, skimage, cc3d, pyqtgraph, tqdm, matplotlib
from PyQt5 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
print("[verify] cv2", cv2.__version__, "PyQt5", QtCore.QT_VERSION_STR, "h5py OK")
PY
echo "[verify] PASS"
