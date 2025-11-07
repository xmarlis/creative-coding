import numpy as np
import matplotlib.pyplot as plt
import argparse
try:
    import warnings
    try:
        from numba import jit
        NUMBA_AVAILABLE = True
    except Exception:
        # fallback: provide a no-op jit decorator so the script runs without numba installed
        NUMBA_AVAILABLE = False
        warnings.warn("Numba is not installed. Running without JIT optimizations. Install numba for much better performance (pip install numba).")
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
except ImportError:
    pass

from matplotlib.colors import LinearSegmentedColormap
import os

@jit(nopython=True)
def burning_ship_classic(c_real, c_imag, max_iter):
    """Classic Burning Ship fractal algorithm."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        z_real_temp = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
        z_imag = 2 * z_real_abs * z_imag_abs + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_cubic(c_real, c_imag, max_iter):
    """Cubic Burning Ship variation."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        r2 = z_real_abs * z_real_abs + z_imag_abs * z_imag_abs
        if r2 == 0:
            z_real = c_real
            z_imag = c_imag
        else:
            theta = np.arctan2(z_imag_abs, z_real_abs)
            r = np.sqrt(r2)
            z_real = r * r * r * np.cos(3 * theta) + c_real
            z_imag = r * r * r * np.sin(3 * theta) + c_imag
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_inverted(c_real, c_imag, max_iter):
    """Inverted Burning Ship - inverting the z values after absolute value."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        r2 = z_real_abs * z_real_abs + z_imag_abs * z_imag_abs
        if r2 < 0.0001:
            z_real_abs, z_imag_abs = 0.0001, 0.0001
        else:
            z_real_abs = z_real_abs / r2
            z_imag_abs = z_imag_abs / r2
        z_real_temp = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
        z_imag = 2 * z_real_abs * z_imag_abs + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_sine(c_real, c_imag, max_iter):
    """Sine Burning Ship - applying sine function to the absolute values."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        z_real_abs, z_imag_abs = np.sin(z_real_abs), np.sin(z_imag_abs)
        z_real_temp = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
        z_imag = 2 * z_real_abs * z_imag_abs + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_power(c_real, c_imag, max_iter, power=2.5):
    """Power Burning Ship - raising to a custom power."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        r = np.sqrt(z_real_abs * z_real_abs + z_imag_abs * z_imag_abs)
        if r == 0:
            z_real = c_real
            z_imag = c_imag
        else:
            theta = np.arctan2(z_imag_abs, z_real_abs)
            r_pow = r ** power
            z_real = r_pow * np.cos(power * theta) + c_real
            z_imag = r_pow * np.sin(power * theta) + c_imag
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_tricorn(c_real, c_imag, max_iter):
    """Tricorn variation - using complex conjugate."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_imag = -z_imag
        z_real_temp = z_real * z_real - z_imag * z_imag + c_real
        z_imag = 2 * z_real * z_imag + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_magnet(c_real, c_imag, max_iter):
    """Burning Ship with magnet-type formula."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs, z_imag_abs = abs(z_real), abs(z_imag)
        denom = z_real_abs * z_real_abs + z_imag_abs * z_imag_abs
        if denom < 0.0001:
            denom = 0.0001
        z_real_temp = (z_real_abs * z_real_abs - z_imag_abs * z_imag_abs) / denom + c_real
        z_imag = (2 * z_real_abs * z_imag_abs) / denom + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_celtic(c_real, c_imag, max_iter):
    """Celtic Burning Ship - absolute value applied to real part only."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_abs = abs(z_real)
        z_real_temp = z_real_abs * z_real_abs - z_imag * z_imag + c_real
        z_imag = 2 * z_real_abs * z_imag + c_imag
        z_real = z_real_temp
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

@jit(nopython=True)
def burning_ship_buffalo(c_real, c_imag, max_iter):
    """Buffalo Burning Ship - absolute values on both components after multiplication."""
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real_temp = z_real * z_real - z_imag * z_imag
        z_imag = 2 * abs(z_real * z_imag)
        z_real = abs(z_real_temp)
        z_real += c_real
        z_imag += c_imag
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter

def create_fractal(h, w, x_min, x_max, y_min, y_max, max_iter, fractal_func, **kwargs):
    """Generate the fractal image."""
    y, x = np.ogrid[y_min:y_max:h*1j, x_min:x_max:w*1j]
    c_real, c_imag = x.reshape(w), y.reshape(h)
    fractal = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            fractal[i, j] = fractal_func(c_real[j], c_imag[i], max_iter, **kwargs)
    return fractal

def get_colormap(colormap_name, max_iter):
    """Generate a colormap based on the provided name."""
    if colormap_name == "fire":
        # Intense fire with deep reds, oranges, and bright yellows
        colors = [(0, 0, 0), (0.2, 0, 0.1), (0.8, 0, 0), (1, 0.3, 0), (1, 0.7, 0), (1, 1, 0.3), (1, 1, 0.8)]
        return LinearSegmentedColormap.from_list("fire_cmap", colors, N=max_iter)
    elif colormap_name == "ocean":
        # Deep ocean with turquoise and cyan highlights
        colors = [(0, 0, 0.1), (0, 0.1, 0.3), (0, 0.3, 0.6), (0, 0.6, 0.8), (0.2, 0.8, 1), (0.5, 1, 1), (0.8, 1, 1)]
        return LinearSegmentedColormap.from_list("ocean_cmap", colors, N=max_iter)
    elif colormap_name == "forest":
        # Rich forest greens with golden highlights
        colors = [(0, 0, 0), (0, 0.15, 0.05), (0, 0.4, 0.1), (0.2, 0.6, 0.2), (0.5, 0.8, 0.3), (0.8, 1, 0.5), (1, 1, 0.7)]
        return LinearSegmentedColormap.from_list("forest_cmap", colors, N=max_iter)
    elif colormap_name == "psychedelic":
        # Vibrant neon colors for eye-catching visuals
        colors = [(0, 0, 0), (1, 0, 0.5), (1, 0, 1), (0.5, 0, 1), (0, 0.5, 1), (0, 1, 1), (0, 1, 0.5), (1, 1, 0)]
        return LinearSegmentedColormap.from_list("psychedelic_cmap", colors, N=max_iter)
    elif colormap_name == "grayscale":
        # High-contrast grayscale
        colors = [(0, 0, 0), (0.3, 0.3, 0.3), (0.7, 0.7, 0.7), (1, 1, 1)]
        return LinearSegmentedColormap.from_list("grayscale_cmap", colors, N=max_iter)
    elif colormap_name == "binary":
        # Vibrant cyan for striking contrast
        vibrant_color = (0, 1, 0.8)
        colors = [(0, 0, 0), vibrant_color]
        return LinearSegmentedColormap.from_list("binary_cmap", colors, N=2)
    # Default: Electric blue to purple gradient
    colors = [(0, 0, 0), (0.1, 0, 0.3), (0.3, 0, 0.8), (0, 0.5, 1), (0.5, 1, 1), (1, 1, 0.5), (1, 0.5, 0)]
    return LinearSegmentedColormap.from_list("burning_ship_cmap", colors, N=max_iter)

def apply_coloring_algorithm(fractal, max_iter, algorithm="standard"):
    """Apply different coloring algorithms to the fractal."""
    colored = np.copy(fractal)
    if algorithm == "smooth":
        mask = colored < max_iter
        colored[mask] = colored[mask] - np.log2(np.log2(colored[mask] + 1)) + 4
    elif algorithm == "histogram":
        mask = colored < max_iter
        hist, _ = np.histogram(colored[mask], bins=max_iter//2, range=(0, max_iter))
        hist = hist.cumsum()
        if hist[-1] > 0:
            hist = hist * max_iter / hist[-1]
            for i in range(len(hist)):
                colored[colored == i] = hist[i]
    elif algorithm == "orbit_trap":
        colored = max_iter - colored
        colored = np.sqrt(colored / max_iter) * max_iter
    elif algorithm == "stripe":
        colored = colored % 16
    return colored

def main():
    parser = argparse.ArgumentParser(description='Generate Burning Ship fractal variations.')
    parser.add_argument('--variation', '-v', 
                        choices=['classic', 'cubic', 'inverted', 'sine', 'power', 
                                 'tricorn', 'magnet', 'celtic', 'buffalo'],
                        default='classic', help='Fractal variation to generate')
    parser.add_argument('--width', '-W', type=int, default=1200, help='Image width (pixels)')
    parser.add_argument('--height', '-H', type=int, default=800, help='Image height (pixels)')
    parser.add_argument('--iterations', '-i', type=int, default=300, help='Maximum iterations')
    parser.add_argument('--power', '-p', type=float, default=2.5, help='Power parameter for power variation')
    parser.add_argument('--output', '-o', default='burning_ship_variation.png', help='Output file name (if not batch/all)')
    parser.add_argument('--folder', '-f', default='fractal_images', help='Folder to save images in')
    parser.add_argument('--colormap', '-c', 
                        choices=['default', 'fire', 'ocean', 'forest', 'psychedelic', 'grayscale', 'binary'],
                        default='fire', help='Color scheme to use')
    parser.add_argument('--coloring', '-C',
                        choices=['standard', 'smooth', 'histogram', 'orbit_trap', 'stripe'],
                        default='smooth', help='Coloring algorithm to apply')
    parser.add_argument('--all', action='store_true', default=True,
                        help='Generate all available variations (default: True)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Save all colormap × coloring combinations (capped by max-images)')
    parser.add_argument('--max-images', type=int, default=15,
                        help='Maximum number of images to generate (global cap)')
     
    args = parser.parse_args()
    
    # base output folder next to this script (single folder, no extra subfolders)
    output_base = os.path.join(os.path.dirname(__file__), args.folder)
    os.makedirs(output_base, exist_ok=True)
    
    # available choices
    colormap_choices = ['default', 'fire', 'ocean', 'forest', 'psychedelic', 'grayscale', 'binary']
    coloring_choices = ['standard', 'smooth', 'histogram', 'orbit_trap', 'stripe']
    variation_list = ['classic', 'cubic', 'inverted', 'sine', 'power', 'tricorn', 'magnet', 'celtic', 'buffalo']
    
    def get_variation_settings(variation):
        # returns x_min, x_max, y_min, y_max, fractal_func, kwargs
        if variation == 'classic':
            return -2.0, 1.0, -2.0, 1.0, burning_ship_classic, {}
        if variation == 'cubic':
            return -1.5, 1.5, -1.5, 1.5, burning_ship_cubic, {}
        if variation == 'inverted':
            return -2.0, 2.0, -2.0, 2.0, burning_ship_inverted, {}
        if variation == 'sine':
            return -3.0, 3.0, -3.0, 3.0, burning_ship_sine, {}
        if variation == 'power':
            return -2.0, 2.0, -2.0, 2.0, burning_ship_power, {'power': args.power}
        if variation == 'tricorn':
            return -2.0, 2.0, -2.0, 2.0, burning_ship_tricorn, {}
        if variation == 'magnet':
            return -2.0, 2.0, -2.0, 2.0, burning_ship_magnet, {}
        if variation == 'celtic':
            return -2.0, 2.0, -2.0, 2.0, burning_ship_celtic, {}
        # buffalo
        return -2.5, 1.5, -2.0, 2.0, burning_ship_buffalo, {}

    # Curated zoom regions with visually interesting perspectives
    zoom_regions = {
        'classic': [
            ("full_view", -2.0, 1.0, -2.0, 1.0),  # Complete classic Burning Ship
            ("ship_bow", -1.88, -1.72, -0.08, 0.08),  # Detailed ship bow
            ("fractal_spires", -1.78, -1.74, -0.02, 0.02),  # Intricate detail
            ("cascade", -1.65, -1.55, 0.025, 0.085),  # Beautiful cascading structures
        ],
        'cubic': [
            ("full_view", -1.5, 1.5, -1.5, 1.5),  # Complete cubic view
            ("trident", -0.3, 0.3, -0.3, 0.3),  # Three-pronged structure
            ("mandala", -0.15, 0.15, -0.15, 0.15),  # Mandala-like center
        ],
        'inverted': [
            ("full_view", -2.0, 2.0, -2.0, 2.0),  # Complete inverted view
            ("vortex", -0.5, 0.5, -0.5, 0.5),  # Swirling vortex
            ("crystalline", -0.2, 0.2, -0.2, 0.2),  # Crystal-like formations
        ],
        'sine': [
            ("full_view", -3.0, 3.0, -3.0, 3.0),  # Complete sine view
            ("ripples", -2.0, 2.0, -2.0, 2.0),  # Wavy ripple patterns
        ],
        'power': [
            ("full_view", -2.0, 2.0, -2.0, 2.0),  # Complete power view
            ("tentacles", -1.2, 1.2, -1.2, 1.2),  # Reaching tendrils
            ("bloom", -0.6, 0.6, -0.6, 0.6),  # Flower-like center
        ],
        'tricorn': [
            ("full_view", -2.0, 2.0, -2.0, 2.0),  # Complete tricorn view
            ("wings", -1.5, 1.5, -1.5, 1.5),  # Butterfly-wing symmetry
        ],
        'magnet': [
            ("full_view", -2.0, 2.0, -2.0, 2.0),  # Complete magnet view
            ("attraction", -1.5, 1.5, -1.5, 1.5),  # Magnetic field lines
        ],
        'celtic': [
            ("full_view", -2.0, 2.0, -2.0, 2.0),  # Complete celtic view
            ("knots", -1.5, 1.5, -1.5, 1.5),  # Celtic knot patterns
        ],
        'buffalo': [
            ("full_view", -2.5, 1.5, -2.0, 2.0),  # Complete buffalo view
            ("horns", -2.0, 1.0, -1.5, 1.5),  # Horn-like structures
        ],
    }

    def get_regions_for_variation(variation):
        if variation in zoom_regions:
            return zoom_regions[variation]
        # fallback: only full view
        x_min, x_max, y_min, y_max, _, _ = get_variation_settings(variation)
        return [("full", x_min, x_max, y_min, y_max)]

    # Curated selection for creative coding - mix of artistic and basic versions
    # This ensures we get 15 high-quality, visually distinct images
    creative_selection = [
        # Basic/clean versions (educational, classic aesthetics)
        ('classic', 'full_view', 'grayscale', 'standard'),
        ('classic', 'ship_bow', 'binary', 'standard'),
        ('cubic', 'full_view', 'grayscale', 'standard'),
        ('buffalo', 'full_view', 'binary', 'standard'),
        ('celtic', 'full_view', 'ocean', 'standard'),
        
        # Artistic versions (vibrant, eye-catching)
        ('classic', 'fractal_spires', 'fire', 'smooth'),
        ('classic', 'cascade', 'psychedelic', 'smooth'),
        ('cubic', 'mandala', 'forest', 'smooth'),
        ('inverted', 'vortex', 'psychedelic', 'orbit_trap'),
        ('inverted', 'crystalline', 'ocean', 'smooth'),
        ('power', 'tentacles', 'fire', 'smooth'),
        ('buffalo', 'horns', 'psychedelic', 'stripe'),
        ('magnet', 'attraction', 'fire', 'orbit_trap'),
        ('sine', 'ripples', 'psychedelic', 'stripe'),
        ('tricorn', 'wings', 'ocean', 'histogram'),
    ]

    # decide which variations to generate
    if args.all and not args.batch:
        # Use curated creative selection (max 15)
        image_count = 0
        for var, region, cmap, coloring in creative_selection[:args.max_images]:
            x_min, x_max, y_min, y_max, fractal_func, kwargs = get_variation_settings(var)
            
            # Get the specific region
            regions = get_regions_for_variation(var)
            region_data = next((r for r in regions if r[0] == region), regions[0])
            _, x_min, x_max, y_min, y_max = region_data
            
            print(f"Generating {var} ({region}) with {cmap} colormap and {coloring} coloring...")
            fractal = create_fractal(args.height, args.width, x_min, x_max, y_min, y_max,
                                   args.iterations, fractal_func, **kwargs)
            
            colored_fractal = apply_coloring_algorithm(fractal, args.iterations, coloring)
            cmap_obj = get_colormap(cmap, args.iterations)
            
            filename = f"{var}_{region}"
            if var == 'power':
                filename += f"_pow{args.power}"
            filename += f"_{cmap}_{coloring}.png"
            output_path = os.path.join(output_base, filename)

            fig = plt.figure(figsize=(10, 8), frameon=False)
            fig.patch.set_facecolor('black')
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.set_facecolor('black')
            fig.add_axes(ax)
            ax.imshow(colored_fractal, cmap=cmap_obj, aspect='equal')
            fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig)

            print(f"Saved to {output_path}")
            image_count += 1
    else:
        # Original logic for single variation or batch mode
        to_generate = variation_list if args.all else [args.variation]
        last_colored_fractal = None
        last_cmap_obj = None
        image_count = 0

        for var in to_generate:
            if image_count >= args.max_images:
                break

            x_min_base, x_max_base, y_min_base, y_max_base, fractal_func, kwargs = get_variation_settings(var)
            regions = get_regions_for_variation(var)

            for region_name, x_min, x_max, y_min, y_max in regions:
                if image_count >= args.max_images:
                    break

                print(f"Generating {var} ({region_name})...")
                fractal = create_fractal(args.height, args.width, x_min, x_max, y_min, y_max,
                                         args.iterations, fractal_func, **kwargs)

                # batch = many colormap × coloring combinations, but still capped by max-images
                if args.batch:
                    for cmap_name in colormap_choices:
                        if image_count >= args.max_images:
                            break
                        for coloring_name in coloring_choices:
                            if image_count >= args.max_images:
                                break
                            colored_fractal = apply_coloring_algorithm(fractal, args.iterations, coloring_name)
                            cmap_obj = get_colormap(cmap_name if cmap_name != 'default' else args.colormap,
                                                    args.iterations)
                            filename = f"{var}_{region_name}"
                            if var == 'power':
                                filename += f"_pow{args.power}"
                            filename += f"_{cmap_name}_{coloring_name}.png"
                            output_path = os.path.join(output_base, filename)

                            fig = plt.figure(figsize=(10, 8), frameon=False)
                            fig.patch.set_facecolor('black')
                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            ax.set_facecolor('black')
                            fig.add_axes(ax)
                            ax.imshow(colored_fractal, cmap=cmap_obj, aspect='equal')
                            fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
                            plt.close(fig)

                            print(f"Saved to {output_path}")
                            last_colored_fractal = colored_fractal
                            last_cmap_obj = cmap_obj
                            image_count += 1
                else:
                    # single combination for this region
                    colored_fractal = apply_coloring_algorithm(fractal, args.iterations, args.coloring)
                    cmap_obj = get_colormap(args.colormap, args.iterations)
                    if args.output == 'burning_ship_variation.png':
                        base_name = f"{var}_{region_name}"
                        if var == 'power':
                            base_name += f"_pow{args.power}"
                        base_name += f"_{args.colormap}_{args.coloring}"
                        output_filename = f"{base_name}.png"
                    else:
                        output_filename = args.output
                    output_path = os.path.join(output_base, output_filename)

                    fig = plt.figure(figsize=(10, 8), frameon=False)
                    fig.patch.set_facecolor('black')
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    ax.set_facecolor('black')
                    fig.add_axes(ax)
                    ax.imshow(colored_fractal, cmap=cmap_obj, aspect='equal')
                    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
                    plt.close(fig)

                    print(f"Saved to {output_path}")
                    last_colored_fractal = colored_fractal
                    last_cmap_obj = cmap_obj
                    image_count += 1

        # only show final image if not batch/all (avoid many popups), without title
        if (not args.batch) and (not args.all) and last_colored_fractal is not None:
            plt.figure(figsize=(10, 8))
            plt.gca().set_facecolor('black')
            plt.imshow(last_colored_fractal, cmap=last_cmap_obj)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()