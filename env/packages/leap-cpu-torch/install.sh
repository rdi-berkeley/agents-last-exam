#!/usr/bin/env bash
# LEAP 1.26 (LLNL CT projector, CPU build) + CPU PyTorch stack into SYSTEM python3.10.
# The CT tasks (ct_geometry_calibration_catphan, limited_angle_ct_dps_reconstruction) have
# software/ wrappers that `exec /usr/bin/python` and `import leaptorch, torch` directly, so
# these must be importable from the system interpreter (not a venv).
#
# WHY VENDORED: there is no installable CPU build of LEAP from upstream. The GitHub release
# libleapct.so is a CUDA build (needs libcufft.so.11); the supported pip/setup.py build runs
# cmake with a CMakeLists that `find_package(CUDA REQUIRED)`; and the alternate
# cpu_CMakeLists.txt is broken (ungated GPU symbols in filtered_backprojection.cpp). So we
# vendor the reference CPU libleapct.so (LEAP 1.26, MIT-licensed) + its pure-Python wrappers
# — the exact CPU artifact used to generate the tasks' reference fixtures. Verified CPU-only
# (ldd has no CUDA linkage) and functional on the lean base.
set -euo pipefail
PY=/usr/bin/python3.10
SP="$($PY -c 'import site; print(site.getsitepackages()[0])')"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP() { uv pip install --python "$PY" --break-system-packages "$@"; }
# CPU torch + sci stack (versions match the reference dev-VM env)
if ! $PY -c "import torch" 2>/dev/null; then
  PIP "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu || PIP torch --index-url https://download.pytorch.org/whl/cpu
fi
PIP "numpy==2.2.6" "scipy==1.15.3" "scikit-image==0.25.2" "imageio==2.37.3" || PIP numpy scipy scikit-image imageio
# Vendored LEAP 1.26 CPU lib + wrappers
if ! $PY -c "import leaptorch" 2>/dev/null; then
  cp "$HERE/vendor/libleapct.so" "$SP/libleapct.so"
  for f in leaptorch.py leapctype.py leap_filter_sequence.py leap_preprocessing_algorithms.py; do
    cp "$HERE/vendor/$f" "$SP/$f"
  done
fi
# Functional check: import + construct a CPU Projector (proves .so loads & binds torch).
$PY -c "import torch,leaptorch; from leaptorch import Projector; Projector(use_gpu=False, batch_size=1); print('[pkg leap-cpu-torch] OK torch', torch.__version__, '+ leaptorch CPU')"
