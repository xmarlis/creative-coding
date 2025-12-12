import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from numba import jit
import os

# ======================================================
#   Circle Map Dynamics
# ======================================================

@jit(nopython=True)
def circle_map(theta, omega, K):
    """
    Circle map iteration:
    θ_{n+1} = θ_n + Ω - (K/(2π)) * sin(2π * θ_n) (mod 1)
    """
    return (theta + omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta)) % 1.0


@jit(nopython=True)
def compute_rotation_number(omega, K, n_iterations=1000, n_transient=200):
    """
    Compute the rotation number (winding number)
    """
    theta = 0.5  # Initial condition
    
    # Skip transient
    for _ in range(n_transient):
        theta = circle_map(theta, omega, K)
    
    # Track unwrapped angle
    theta_unwrapped = theta
    for _ in range(n_iterations):
        theta_unwrapped += omega - (K / (2 * np.pi)) * np.sin(2 * np.pi * theta_unwrapped)
    
    return (theta_unwrapped - theta) / n_iterations


@jit(nopython=True)
def compute_arnold_tongues(omega_range, K_range, n_iterations=1000):
    """
    Compute rotation numbers for full parameter grid
    """
    n_omega = len(omega_range)
    n_K = len(K_range)
    rotation_numbers = np.zeros((n_K, n_omega))
    
    for i in range(n_K):
        K = K_range[i]
        for j in range(n_omega):
            omega = omega_range[j]
            rotation_numbers[i, j] = compute_rotation_number(omega, K, n_iterations)
    
    return rotation_numbers


# ======================================================
#   Color Mapping
# ======================================================

def create_colorful_visualization(rotation_numbers):
    """
    Base HSV mapping
    """
    hue = np.mod(rotation_numbers * 12, 1.0)
    saturation = np.ones_like(hue) * 0.95
    value = np.ones_like(hue) * 0.95
    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv_to_rgb(hsv)


# ======================================================
#   Main Parameters
# ======================================================

omega_min, omega_max = 0.0, 1.0
K_min, K_max = 0.0, 2.0
n_omega = 800
n_K = 600

print("Computing Arnold tongues...")
print(f"Grid: {n_omega} x {n_K}")

omega_range = np.linspace(omega_min, omega_max, n_omega)
K_range = np.linspace(K_min, K_max, n_K)

rotation_numbers = compute_arnold_tongues(omega_range, K_range, n_iterations=1500)

# Debug information
print(f"Rotation number range: [{rotation_numbers.min():.4f}, {rotation_numbers.max():.4f}]")
print(f"Mean: {rotation_numbers.mean():.4f}")

script_dir = os.path.dirname(os.path.abspath(__file__))


# ======================================================
#   Multiple Color Schemes
# ======================================================

color_modes = {
    "normal": lambda R: hsv_to_rgb(np.stack([
        np.mod(R * 12, 1.0),
        0.95 * np.ones_like(R),
        0.95 * np.ones_like(R)
    ], axis=-1)),

    "bands": lambda R: hsv_to_rgb(np.stack([
        np.mod(np.round(R * 20) / 20, 1.0),
        np.ones_like(R),
        np.ones_like(R)
    ], axis=-1)),

    "psychedelic": lambda R: hsv_to_rgb(np.stack([
        np.mod(R * 40, 1.0),
        0.8 + 0.2 * np.sin(10 * R),
        0.8 + 0.2 * np.cos(10 * R)
    ], axis=-1))
}

# Resolutions for output
resolutions = [150, 300, 500]

print("\nCreating multiple Arnold tongue versions...\n")

# ======================================================
#   Generate All Variants
# ======================================================

for cname, color_fn in color_modes.items():
    
    rgb_image = color_fn(rotation_numbers)

    for dpi in resolutions:
        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis("off")

        ax.imshow(rgb_image, aspect='auto', origin='lower', interpolation='bilinear')

        filename = f"arnold_{cname}_{dpi}dpi.png"
        output_file = os.path.join(script_dir, filename)

        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()

        print(f"✓ Saved: {filename}")

print("\nAll Arnold tongue versions created successfully!")
