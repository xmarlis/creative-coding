import numpy as np
import matplotlib.pyplot as plt
import os
import random

# -------- Penrose-Tribar --------
def draw_tribar(ax, cx, cy, size, angle, colors,
                edge_alpha=1.0, fill_alpha=0.9):
    """
    Ein Penrose-Tribar aus 3 Rechtecken.
    cx, cy  - Zentrum
    size    - Skalierung
    angle   - Grundrotation
    colors  - Liste aus 3 Farben (hex oder RGBA)
    """
    L = 1.3 * size   # Länge
    T = 0.40 * size  # Dicke

    rect = np.array([
        [-L/2, -T/2],
        [ L/2, -T/2],
        [ L/2,  T/2],
        [-L/2,  T/2],
    ])

    base_angles = [0, 2*np.pi/3, 4*np.pi/3]
    base_angles = [a + angle for a in base_angles]

    for k, ang in enumerate(base_angles):
        R = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang),  np.cos(ang)],
        ])
        poly = rect @ R.T

        # leichter Schub nach innen für „unmöglichen“ Look
        poly += np.array([
            -0.22 * size * np.cos(ang),
            -0.22 * size * np.sin(ang)
        ])

        poly[:, 0] += cx
        poly[:, 1] += cy

        ax.fill(
            poly[:, 0], poly[:, 1],
            facecolor=colors[k],
            edgecolor=(1, 1, 1, edge_alpha),
            linewidth=0.5,
            joinstyle="round",
            alpha=fill_alpha
        )

# -------- Ausgabe-Datei im gleichen Ordner --------
script_dir = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(script_dir, "penrose_random_field_multiscale.png")

# -------- Figur / Canvas --------
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

# Farbpaletten (Neon + Pastell gemischt)
palette = [
    "#ff0080", "#ff6600", "#ffcc00",
    "#00ffcc", "#00ccff", "#9966ff",
    "#ffb3ba", "#baffc9", "#bae1ff"
]

def random_colors():
    # 3 zufällige Farben aus Palette
    return random.sample(palette, 3)

# -------- verschiedene Größen-Ebenen --------
random.seed(42)
np.random.seed(42)

# (Anzahl, min_size, max_size, radius)
layers = [
    (10, 1.5, 2.4, 6.0),   # wenige große Formen im Zentrum
    (35, 0.7, 1.2, 8.0),   # mittlere
    (80, 0.25, 0.6, 9.0),  # viele kleine außen
]

for n, smin, smax, R in layers:
    for _ in range(n):
        # zufällige Position in Kreis mit Radius R
        r = np.sqrt(np.random.uniform(0, R**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        size = np.random.uniform(smin, smax)
        angle = np.random.uniform(0, 2*np.pi)
        cols = random_colors()

        # große Formen etwas transparenter, damit sich Ebenen überlagern
        alpha = 0.9 if size < 1.2 else 0.7

        draw_tribar(ax, x, y, size, angle, cols,
                    edge_alpha=alpha, fill_alpha=alpha)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.savefig(outfile, bbox_inches="tight", facecolor="black")
plt.close()
print("Saved:", outfile)
