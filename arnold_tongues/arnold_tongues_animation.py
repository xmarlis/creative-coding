import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import hsv_to_rgb
from numba import jit
import os

# ============================================
#  Circle Map Dynamics
# ============================================

@jit(nopython=True)
def circle_map(theta, omega, K):
    return (theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)) % 1.0

@jit(nopython=True)
def compute_rotation_number(omega, K, n_iterations=1000, n_transient=200):
    theta = 0.5
    for _ in range(n_transient):
        theta = circle_map(theta, omega, K)

    theta_unwrapped = theta
    for _ in range(n_iterations):
        theta_unwrapped += omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_unwrapped)

    return (theta_unwrapped - theta) / n_iterations

@jit(nopython=True)
def compute_grid(omega_range, K, n_iterations=1000):
    """
    Computes rotation numbers for a single K.
    """
    n = len(omega_range)
    result = np.zeros(n)
    for i in range(n):
        result[i] = compute_rotation_number(omega_range[i], K, n_iterations)
    return result


# ============================================
#  Color Mapping
# ============================================

def colorize(rotation_numbers):
    hue = np.mod(rotation_numbers * 12, 1.0)
    sat = np.ones_like(hue) * 0.95
    val = np.ones_like(hue) * 0.95
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


# ============================================
#  Animation Setup
# ============================================

omega_range = np.linspace(0.0, 1.0, 900)
K_values = np.linspace(0.0, 2.0, 240)  # 240 frames

fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

img = ax.imshow(
    np.zeros((1, len(omega_range), 3)),
    aspect='auto',
    origin='lower',
    interpolation='bilinear'
)


# ============================================
#  Frame Update Function
# ============================================

def update(frame):
    K = K_values[frame]
    print(f"Rendering frame {frame+1}/{len(K_values)}  (K={K:.3f})")

    rot = compute_grid(omega_range, K, n_iterations=1200)
    
    rgb = colorize(rot)[np.newaxis, :, :]  # reshape to 2D image
    
    img.set_data(rgb)
    return [img]


# ============================================
#  Create Animation
# ============================================

script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "arnold_tongues_animation.mp4")
writer = FFMpegWriter(fps=30, bitrate=5000)

print("\nCreating animation...\n")

anim = FuncAnimation(
    fig,
    update,
    frames=len(K_values),
    interval=30,
    blit=True
)

anim.save(output_file, writer=writer, savefig_kwargs={'pad_inches': 0})

print(f"\n✓ Animation saved as: {output_file}")
print("Done!")
