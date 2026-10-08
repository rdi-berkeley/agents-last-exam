"""Check the staged QA environment; install pinned packages only with --install."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

PINS = {
    "pydicom": "3.0.1",
    "pymedphys": "0.41.0",
    "numba": "0.65.0",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "matplotlib": "3.10.8",
    "scikit-image": "0.25.2",
    "Pillow": "12.2.0",
}
REQUIRED_VERSIONS = {name: PINS[name] for name in ("pydicom", "pymedphys", "numba")}
IMPORT_NAMES = {"scikit-image": "skimage", "Pillow": "PIL"}


def probe(reference: str | None = None) -> dict:
    modules = {}
    versions = {}
    for package, version in PINS.items():
        module = importlib.import_module(IMPORT_NAMES.get(package, package))
        versions[package] = module.__version__
        if package in REQUIRED_VERSIONS and versions[package] != version:
            raise RuntimeError(f"{package}: expected {version}, got {versions[package]}")
        modules[package] = module
    pydicom, pymedphys, np = (modules[k] for k in ("pydicom", "pymedphys", "numpy"))
    mode = pydicom.config.settings.reading_validation_mode
    if mode != pydicom.config.WARN:
        raise RuntimeError(f"Unexpected DICOM reading validation mode: {mode}")
    axis = np.arange(4, dtype=float)
    dose = np.arange(1, 5, dtype=float)
    gamma = pymedphys.gamma(
        (axis,), dose, (axis,), dose,
        dose_percent_threshold=3, distance_mm_threshold=3,
    )
    if not np.isfinite(gamma).all() or not np.allclose(gamma, 0):
        raise RuntimeError("pymedphys gamma identity check failed")
    if reference:
        ds = pydicom.dcmread(reference)
        values = ds.pixel_array.astype(float) * float(ds.DoseGridScaling)
        if not np.isfinite(values).all():
            raise RuntimeError("Reference dose contains non-finite values")
    return {
        "ok": True, "versions": versions, "reading_validation_mode": mode,
        "python": sys.executable, "prefix": sys.prefix,
        "gamma_identity_ok": True, "reference_checked": bool(reference),
    }


def fresh_probe(reference: str | None) -> dict:
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--probe"]
    if reference:
        command.extend(["--reference", reference])
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"ok": False, "error": (result.stderr or result.stdout)[-2000:]}
    if result.returncode != 0:
        report["ok"] = False
    return report


def ensure_runtime(reference: str | None = None, *, install: bool = False) -> dict:
    installed = False
    if install:
        lock_path = Path(sys.prefix) / ".ale_matrad_python.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report = fresh_probe(None)
            if not report.get("ok"):
                print("[matrad runtime] provisioning: " + report.get("error", "probe failed"), flush=True)
                subprocess.run(
                    [sys.executable, "-I", "-m", "ensurepip", "--upgrade"], check=True, timeout=120,
                )
                subprocess.run(
                    [sys.executable, "-I", "-m", "pip", "--isolated", "install",
                     "--disable-pip-version-check", "--no-input", "--only-binary=:all:",
                     "--index-url", "https://pypi.org/simple", "--retries", "3", "--timeout", "30",
                     "--log", str(Path(sys.prefix) / ".ale_matrad_pip.log"),
                     *[f"{package}=={version}" for package, version in PINS.items()]],
                    check=True, timeout=1200,
                )
                installed = True
    report = fresh_probe(reference)
    if not report.get("ok"):
        raise RuntimeError(
            "QA preflight failed. Check the environment and reference input; "
            "use ensure_runtime.py --install through software/run_matrad.sh only to "
            "provision missing dependencies before starting the task: " + json.dumps(report)
        )
    return {**report, "installed": installed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--install", action="store_true", help="Provision dependencies before evaluation")
    args = parser.parse_args()
    try:
        result = probe(args.reference) if args.probe else ensure_runtime(args.reference, install=args.install)
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if args.report:
        temp = args.report.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2))
        os.replace(temp, args.report)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result.get("ok") else 1)
