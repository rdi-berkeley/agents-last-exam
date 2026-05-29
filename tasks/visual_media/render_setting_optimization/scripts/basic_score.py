"""VM-side scoring script for render_setting_optimization.

Compares two EXR renders (seeded with different random seeds but identical
settings) to measure convergence quality.  Lower noise between the two
renders yields a higher score.

Dependencies (must be available on the MoonRay VM):
  - OpenEXR
  - Imath
  - numpy

    Usage:
    python basic_score.py seed_A.exr seed_B.exr

Output:
    Final_Score: X.XXXX
"""

import sys

import Imath
import numpy as np
import OpenEXR


def read_exr(filepath):
    """Load an EXR file and return an (H, W, 3) float32 RGB array."""
    file = OpenEXR.InputFile(filepath)
    dw = file.header()["displayWindow"]
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    data = []
    all_channels = file.header()["channels"].keys()
    for c in ["R", "G", "B"]:
        target = c
        if c not in all_channels:
            found = [ac for ac in all_channels if ac.endswith("." + c)]
            if found:
                target = found[0]
        channel_data = np.frombuffer(file.channel(target, pt), dtype=np.float32)
        data.append(channel_data.reshape(size))
    return np.stack(data, axis=-1)


def calculate_benchmark(file_a, file_b):
    """Return a quality score in [0.0, 1.0] for a pair of seeded renders."""
    img_a = read_exr(file_a)
    img_b = read_exr(file_b)

    # 1. Relative uncertainty (noise)
    noise = np.abs(img_a - img_b) / np.sqrt(2)

    # 2. Signal (average brightness + epsilon)
    signal = (np.abs(img_a) + np.abs(img_b)) / 2.0 + 1e-4

    # 3. Coefficient of variation
    cv = noise / signal

    # 4. Exponential mapping (alpha=1.0: 10% relative error ~ 0.90 score)
    alpha = 1.0
    pixel_scores = np.exp(-alpha * cv)

    # 5. Final score
    base_score = np.mean(pixel_scores)
    return base_score


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python basic_score.py seed_A.exr seed_B.exr")
        sys.exit(1)

    score = calculate_benchmark(sys.argv[1], sys.argv[2])
    print(f"Final_Score: {score:.4f}")
