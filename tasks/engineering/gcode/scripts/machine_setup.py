"""Declare the unloaded starting state for native CAM program replay."""

import hashlib
import json
import math
from pathlib import Path


def prepare_program(program, output, *, stock_top_mm, clearance_mm):
    for value in (stock_top_mm, clearance_mm):
        if not math.isfinite(value):
            raise ValueError("Machine setup requires finite original stock and clearance values")
    if clearance_mm <= 0:
        raise ValueError("Initial clearance must be above the original stock")
    program, output = Path(program), Path(output)
    if program.resolve() == output.resolve():
        raise ValueError("Machine initialization must not overwrite the exported program")
    initial_z = stock_top_mm + clearance_mm
    if not math.isfinite(initial_z):
        raise ValueError("Initial machine position must be finite")
    digest = hashlib.sha256()
    with program.open("rb") as source, output.open("xb") as destination:
        destination.write(
            f"(Unloaded machine initialization before the saved CAM job)\n"
            f"G21 G90 G53 G0 Z{initial_z:.12f}\n".encode()
        )
        for line in source:
            if line.strip() == b"%":
                raise ValueError("Native exported program must not contain tape delimiters")
            digest.update(line)
            destination.write(line)
    receipt = {
        "exported_program_sha256": digest.hexdigest(),
        "initial_machine_mm": [0.0, 0.0, 0.0],
        "initial_tool": 0,
        "unloaded_reposition_mm": [0.0, 0.0, initial_z],
        "stock_top_mm": stock_top_mm,
        "clearance_mm": clearance_mm,
        "original_program_preserved": True,
    }
    output.with_suffix(output.suffix + ".setup.json").write_text(json.dumps(receipt, indent=2))
    return receipt
