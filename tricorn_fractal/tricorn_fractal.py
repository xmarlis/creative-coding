import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os
from datetime import datetime

def compute_tricorn(xmin, xmax, ymin, ymax, width, height, max_iterations=256, escape_radius=2.0):
    """
    Compute the Tricorn (Mandelbar) fractal with smooth iteration count.
    The Tricorn uses: z = conj(z)^2 + c (complex conjugate before squaring)
    
    This creates a fractal with real-axis symmetry and intricate filament structures.
    
    Parameters:
    -----------
    xmin, xmax, ymin, ymax : float
        The boundaries of the complex plane to compute
    width, height : int
        The dimensions of the output array
    max_iterations : int
        Maximum iterations before considering a point in the set
    escape_radius : float
        Radius beyond which we consider the point escaped
    
    Returns:
    --------
    numpy.ndarray
        2D array containing smooth iteration counts
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymax, ymin, height)
    xx, yy = np.meshgrid(x, y)
    c = xx + 1j * yy
    
    z = np.zeros_like(c, dtype=complex)
    result = np.zeros(c.shape, dtype=float)
    mask = np.ones(c.shape, dtype=bool)
    
    escape_radius_sq = escape_radius ** 2
    
    for i in range(max_iterations):
        # Tricorn iteration: conjugate before squaring
        z[mask] = np.conj(z[mask]) ** 2 + c[mask]
        
        escaped = (np.abs(z) ** 2 > escape_radius_sq) & mask
        
        if np.any(escaped):
            z_escaped = z[escaped]
            abs_z = np.abs(z_escaped)
            result[escaped] = i + 1 - np.log2(np.log2(abs_z + 1e-10))
            mask[escaped] = False
        
        if not np.any(mask):
            break
    
    # Store final z values for advanced coloring
    z_final = z.copy()
    
    result[mask] = max_iterations
    return result, z_final

def get_default_views():
    """
    Get interesting viewing windows for the Tricorn fractal.
    """
    return {
        "full": {"xmin": -2.5, "xmax": 1.5, "ymin": -2.0, "ymax": 2.0},
        "tip": {"xmin": -0.8, "xmax": -0.4, "ymin": -0.3, "ymax": 0.1},
        "filament": {"xmin": -0.65, "xmax": -0.55, "ymin": -0.05, "ymax": 0.05},
        "spiral": {"xmin": -0.62, "xmax": -0.58, "ymin": -0.02, "ymax": 0.02},
        "deep": {"xmin": -0.605, "xmax": -0.595, "ymin": -0.005, "ymax": 0.005}
    }

def save_figure_clean(fig, out_dir, prefix=None, dpi=150):
    """Save figure with anonymous timestamp filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{prefix}_{timestamp}.png" if prefix else f"{timestamp}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    return out_path

def colorize_fire_ice(fractal, z_final, max_iter):
    """Fire and ice dual-tone coloring."""
    norm = fractal / max_iter
    
    # Fire palette (hot colors)
    fire_r = np.clip(norm * 3.5, 0, 1)
    fire_g = np.clip((norm - 0.3) * 2.5, 0, 1)
    fire_b = np.clip((norm - 0.6) * 2.0, 0, 0.3)
    
    # Ice palette (cool colors)
    ice_r = np.clip((1 - norm) * 0.4, 0, 1)
    ice_g = np.clip((1 - norm) * 1.2, 0, 1)
    ice_b = np.clip((1 - norm) * 1.5, 0, 1)
    
    # Mix based on angle
    angle = (np.angle(z_final) + np.pi) / (2 * np.pi)
    mix = angle
    
    r = fire_r * mix + ice_r * (1 - mix)
    g = fire_g * mix + ice_g * (1 - mix)
    b = fire_b * mix + ice_b * (1 - mix)
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb ** 0.85, 0, 1)

def colorize_electric(fractal, z_final, max_iter):
    """Electric neon glow effect."""
    norm = fractal / max_iter
    angle = np.angle(z_final)
    
    h = ((angle + np.pi) / (2 * np.pi) + norm * 2.5) % 1.0
    s = 0.9 - 0.3 * norm
    v = 0.2 + 0.8 * ((1 - norm) ** 0.4)
    
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # Add electric glow
    glow = (1 - norm) ** 2
    rgb = rgb + 0.15 * glow[..., None] * np.array([0.3, 0.7, 1.0])
    
    return np.clip(rgb ** 0.8, 0, 1)

def colorize_cosmic(fractal, z_final, max_iter):
    """Deep space cosmic colors."""
    norm = fractal / max_iter
    
    # Deep space base
    r = 0.05 + 0.4 * np.sin(norm * 8.0) ** 2
    g = 0.02 + 0.3 * np.sin(norm * 6.0 + 1.0) ** 2
    b = 0.15 + 0.6 * np.sin(norm * 10.0 + 2.0) ** 2
    
    # Star field highlights
    stars = ((1 - norm) ** 4) * 0.8
    r += stars
    g += stars * 0.9
    b += stars * 0.95
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def colorize_rainbow_swirl(fractal, z_final, max_iter):
    """Rainbow spiral patterns."""
    norm = fractal / max_iter
    angle = np.angle(z_final)
    
    h = ((angle + np.pi) / (2 * np.pi) * 3.0 + norm * 5.0) % 1.0
    s = 0.7 + 0.3 * np.sin(norm * 12.0)
    v = 0.3 + 0.7 * (1 - norm) ** 0.6
    
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return np.clip(rgb ** 0.9, 0, 1)

def colorize_ocean_depth(fractal, z_final, max_iter):
    """Ocean depth gradient with turquoise and deep blue."""
    norm = fractal / max_iter
    
    # Turquoise to deep blue gradient
    r = 0.05 + 0.3 * (1 - norm) ** 2
    g = 0.3 + 0.5 * (1 - norm) ** 0.8
    b = 0.5 + 0.5 * (1 - norm) ** 0.5
    
    # Add wave patterns
    wave = 0.15 * np.sin(norm * 15.0 + np.real(z_final) * 5.0)
    g += wave
    b += wave * 0.5
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def colorize_sunset(fractal, z_final, max_iter):
    """Warm sunset colors."""
    norm = fractal / max_iter
    angle = (np.angle(z_final) + np.pi) / (2 * np.pi)
    
    # Sunset palette
    r = 0.9 + 0.1 * np.sin(norm * 10.0)
    g = 0.3 + 0.5 * (1 - norm) ** 0.6
    b = 0.1 + 0.3 * (1 - norm) ** 1.5
    
    # Modulate by angle for variation
    r *= 0.8 + 0.4 * angle
    g *= 0.7 + 0.6 * (1 - angle)
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb ** 0.88, 0, 1)

def colorize_aurora(fractal, z_final, max_iter):
    """Aurora borealis effect."""
    norm = fractal / max_iter
    angle = np.angle(z_final)
    
    # Green-purple aurora
    wave1 = np.sin(norm * 8.0 + angle * 2.0)
    wave2 = np.sin(norm * 12.0 - angle * 1.5 + 2.0)
    
    r = 0.2 + 0.6 * np.clip(wave2, 0, 1) * (1 - norm)
    g = 0.4 + 0.6 * np.clip(wave1, 0, 1) * (1 - norm)
    b = 0.3 + 0.7 * np.clip((wave1 + wave2) / 2, 0, 1) * (1 - norm)
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb ** 0.82, 0, 1)

if __name__ == "__main__":
    # create a single timestamped output folder next to this script (no nested folders)
    base_dir = os.path.dirname(__file__)
    run_dir = os.path.join(base_dir, datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    
    out_dir = run_dir  # use run_dir for all outputs
    
    views = get_default_views()
    width, height = 1200, 1200
    max_iter = 512
    
    # Standard colormaps
    colormaps = ['twilight_shifted', 'cubehelix', 'viridis', 'plasma', 'cividis']
    
    # Special colorizers
    special_colorizers = {
        "fire_ice": colorize_fire_ice,
        "electric": colorize_electric,
        "cosmic": colorize_cosmic,
        "rainbow": colorize_rainbow_swirl,
        "ocean": colorize_ocean_depth,
        "sunset": colorize_sunset,
        "aurora": colorize_aurora,
    }
    
    images_created = 0
    max_images = 10
    done = False
    
    for view_name, view in views.items():
        if images_created >= max_images:
            break
        print(f"Computing Tricorn fractal - {view_name} view...")
        fractal, z_final = compute_tricorn(
            view["xmin"], view["xmax"],
            view["ymin"], view["ymax"],
            width, height,
            max_iterations=max_iter
        )
        
        # Standard colormap versions (fewer to make room for special ones)
        for cmap in colormaps[:1]:
            if images_created >= max_images:
                done = True
                break
            fig = plt.figure(figsize=(10, 10), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            gamma = 0.7
            normalized = fractal / max_iter
            enhanced = np.power(normalized, gamma) * max_iter
            
            ax.imshow(enhanced, cmap=cmap, origin='upper')
            
            out_path = save_figure_clean(fig, out_dir, prefix=None, dpi=150)
            plt.close(fig)
            images_created += 1
            print(f"  Saved {view_name}/{cmap}: {out_path} ({images_created}/{max_images})")
        
        if done or images_created >= max_images:
            break
        
        # Special artistic colorizers
        for colorizer_name, colorizer in special_colorizers.items():
            if images_created >= max_images:
                done = True
                break
            fig = plt.figure(figsize=(10, 10), frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            rgb_image = colorizer(fractal, z_final, max_iter)
            ax.imshow(rgb_image, origin='upper')
            
            out_path = save_figure_clean(fig, out_dir, prefix=None, dpi=150)
            plt.close(fig)
            images_created += 1
            print(f"  Saved {view_name}/{colorizer_name}: {out_path} ({images_created}/{max_images})")
        
        if done:
            break
    
    print(f"\nAll Tricorn fractals saved in: {out_dir} (created {images_created} images)")
