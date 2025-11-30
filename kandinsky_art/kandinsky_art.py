import random
import math
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D

# ----------------------
# CONFIG
# ----------------------
IMAGE_SIZE = 8        # inches, square
NUM_SHAPES = 40       # total number of shapes (reduced from 80)
BACKGROUND_COLOR = (0.0, 0.0, 0.0)  # black background

# ----------------------
# HELPERS
# ----------------------
def random_color(pastel=False):
    """Return a random RGB color tuple."""
    r, g, b = random.random(), random.random(), random.random()
    if pastel:
        r = (r + 1.0) / 2.0
        g = (g + 1.0) / 2.0
        b = (b + 1.0) / 2.0
    return (r, g, b)

def random_position():
    """Return a random (x, y) in [0,1]x[0,1]."""
    return random.random(), random.random()

def random_size(min_size=0.08, max_size=0.35):
    return random.uniform(min_size, max_size)

# ----------------------
# MAIN
# ----------------------
def make_kandinsky_art(
    image_size=IMAGE_SIZE,
    num_shapes=NUM_SHAPES,
    background_color=BACKGROUND_COLOR,
    seed=None,
    save_path=None,
):
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots(figsize=(image_size, image_size))

    # FULL BLACK BACKGROUND (important!)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Keep coordinate system in [0,1] x [0,1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    for _ in range(num_shapes):
        shape_type = random.choice(["circle", "rect", "triangle", "line", "arc"])

        x, y = random_position()
        size = random_size()
        color = random_color(pastel=False)
        alpha = random.uniform(0.4, 0.9)

        if shape_type == "circle":
            radius = size * 0.5
            circle = Circle(
                (x, y),
                radius,
                facecolor=color,
                edgecolor=random_color(),
                linewidth=random.uniform(0.5, 2.5),
                alpha=alpha,
            )
            ax.add_patch(circle)

        elif shape_type == "rect":
            w = size * random.uniform(0.4, 1.0)
            h = size * random.uniform(0.4, 1.0)
            rect = Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                angle=random.uniform(-45, 45),
                facecolor=color,
                edgecolor=random_color(),
                linewidth=random.uniform(0.5, 2.5),
                alpha=alpha,
            )
            ax.add_patch(rect)

        elif shape_type == "triangle":
            angle0 = random.random() * 2 * math.pi
            r = size * 0.6
            points = []
            for i in range(3):
                angle = angle0 + i * 2 * math.pi / 3
                px = x + r * math.cos(angle)
                py = y + r * math.sin(angle)
                points.append((px, py))
            tri = Polygon(
                points,
                closed=True,
                facecolor=color,
                edgecolor=random_color(),
                linewidth=random.uniform(0.5, 2.5),
                alpha=alpha,
            )
            ax.add_patch(tri)

        elif shape_type == "line":
            length = size
            angle = random.random() * 2 * math.pi
            x2 = x + length * math.cos(angle)
            y2 = y + length * math.sin(angle)
            line = Line2D(
                [x, x2],
                [y, y2],
                linewidth=random.uniform(1.0, 4.0),
                color=random_color(pastel=False),
                alpha=alpha,
            )
            ax.add_line(line)

        elif shape_type == "arc":
            radius = size * random.uniform(0.4, 0.8)
            start_angle = random.uniform(0, 360)
            end_angle = start_angle + random.uniform(30, 200)
            num_points = 30
            angles = [
                math.radians(start_angle + i * (end_angle - start_angle) / (num_points - 1))
                for i in range(num_points)
            ]
            points_outer = [
                (x + radius * math.cos(a), y + radius * math.sin(a)) for a in angles
            ]
            width = radius * 0.15
            points_inner = [
                (x + (radius - width) * math.cos(a), y + (radius - width) * math.sin(a))
                for a in reversed(angles)
            ]
            arc_points = points_outer + points_inner
            arc = Polygon(
                arc_points,
                closed=True,
                facecolor="none",
                edgecolor=random_color(pastel=False),
                linewidth=random.uniform(1.0, 3.0),
                alpha=alpha,
            )
            ax.add_patch(arc)

    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


if __name__ == "__main__":
    # Create output folder in the same directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"kandinsky_art_{timestamp}.png"

    make_kandinsky_art(seed=None, save_path=str(save_path))

    print(f"Artwork saved to: {save_path}")
