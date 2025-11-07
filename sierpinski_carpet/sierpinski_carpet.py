import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def generate_carpet(depth: int) -> np.ndarray:
    """
    Generate a Sierpinski carpet as a 2D numpy array with values 0 or 1.
    depth = 0 -> single filled square
    depth = 1 -> 3x3 grid with middle removed
    depth = 2 -> 9x9 grid, etc.
    """
    n = 3 ** depth
    carpet = np.ones((n, n), dtype=np.uint8)

    for x in range(n):
        for y in range(n):
            xx, yy = x, y
            filled = 1
            # base-3 test: if we ever hit (1,1) in base-3 digits it's a hole
            while xx > 0 or yy > 0:
                if xx % 3 == 1 and yy % 3 == 1:
                    filled = 0
                    break
                xx //= 3
                yy //= 3
            carpet[y, x] = filled

    return carpet


def save_carpet_image(carpet: np.ndarray, output_path: str):
    """Save the carpet array as a high-res PNG on black/white background."""
    fig = plt.figure(figsize=(6, 6))
    # use the whole figure area (no margins)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(carpet, cmap="gray", interpolation="nearest")
    ax.set_axis_off()

    # tight, no extra borders
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # Create a time-stamped output folder next to this script
    base_dir = os.path.dirname(__file__)
    run_dir = os.path.join(base_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    print("Generating Sierpinski carpet images...")
    # choose which depths you want (1–5 is usually nice)
    for depth in range(1, 6):
        carpet = generate_carpet(depth)
        filename = f"sierpinski_carpet_depth_{depth}.png"
        output_path = os.path.join(run_dir, filename)
        save_carpet_image(carpet, output_path)
        print(f"  saved {filename}")

    print(f"All images saved in: {run_dir}")
