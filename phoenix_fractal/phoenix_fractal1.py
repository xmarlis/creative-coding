import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os

def phoenix_fractal(width=1920, height=1080, max_iter=256, 
                   c_real=-0.5, c_imag=0.0, 
                   p_real=0.56667, p_imag=0.0,
                   zoom=1.0, center_x=0.0, center_y=0.0):
    """Generate a Phoenix fractal."""
    
    print(f"Generating: c={c_real}+{c_imag}i, p={p_real}+{p_imag}i")
    
    # Define the complex plane boundaries
    x_min, x_max = center_x - 1.5/zoom, center_x + 1.5/zoom
    y_min, y_max = center_y - 1.5/zoom * height/width, center_y + 1.5/zoom * height/width
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays
    Z = X + 1j*Y
    Z_prev = np.zeros_like(Z)
    iterations = np.zeros(Z.shape, dtype=int)
    
    # Complex parameters
    c = c_real + 1j*c_imag
    p = p_real + 1j*p_imag
    
    # Create mask for points that haven't escaped
    mask = np.ones(Z.shape, dtype=bool)
    
    # Iterate
    for i in range(max_iter):
        # Phoenix formula: z_{n+1} = z_n^2 + c + p * z_{n-1}
        Z_temp = Z**2 + c + p * Z_prev
        
        # Update for next iteration
        Z_prev = Z
        Z = Z_temp
        
        # Check which points have escaped
        escaped = np.abs(Z) > 10.0
        
        # Record iteration count for newly escaped points
        newly_escaped = escaped & mask
        iterations[newly_escaped] = i
        
        # Update mask
        mask = mask & ~escaped
        
        # Early exit if all points have escaped
        if not mask.any():
            break
    
    # Points that never escaped get max iterations
    iterations[mask] = max_iter
    
    return iterations

def create_colormap(name='fire'):
    """Create various colormaps for different aesthetics"""
    colormaps = {
        'fire': ['#000033', '#1a0066', '#4a0080', '#8000ff', '#ff0080', 
                 '#ff3300', '#ff6600', '#ffaa00', '#ffff00', '#ffffff'],
        'ice': ['#000011', '#001133', '#003366', '#0066cc', '#00aaff',
                '#33ccff', '#66ddff', '#99eeff', '#ccffff', '#ffffff'],
        'emerald': ['#000000', '#001a00', '#003300', '#006600', '#009900',
                    '#00cc00', '#00ff00', '#66ff66', '#ccffcc', '#ffffff'],
        'sunset': ['#000033', '#330033', '#660033', '#990033', '#cc3300',
                   '#ff6600', '#ff9900', '#ffcc00', '#ffff66', '#ffffcc'],
        'ocean': ['#000022', '#000044', '#001166', '#0033aa', '#0066ff',
                  '#3399ff', '#66ccff', '#99ddff', '#ccffff', '#ffffff'],
        'plasma': ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
                   '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
        'volcano': ['#000000', '#1a0000', '#330000', '#660000', '#990000',
                    '#cc0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00'],
        'purple': ['#000000', '#0f000f', '#1f001f', '#4f004f', '#8000ff',
                   '#a000ff', '#c000ff', '#d966ff', '#eeccff', '#ffffff'],
        'gold': ['#000000', '#1a0f00', '#331f00', '#664400', '#997700',
                 '#ccaa00', '#ffcc00', '#ffdd66', '#ffeecc', '#ffffff'],
        'mint': ['#000022', '#001a1a', '#003333', '#006666', '#009999',
                 '#00cccc', '#33ffff', '#99ffff', '#ccffff', '#ffffff']
    }
    
    colors = colormaps.get(name, colormaps['fire'])
    return LinearSegmentedColormap.from_list(name, colors, N=256)

def plot_fractal(iterations, filename, colormap='fire', output_dir=None):
    """Plot and save the fractal"""
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='black')
    
    # Normalize iterations for better color distribution
    norm_iterations = np.sqrt(iterations.astype(float))
    
    cmap = create_colormap(colormap)
    
    im = ax.imshow(norm_iterations, cmap=cmap, interpolation='bilinear',
                   extent=[0, iterations.shape[1], 0, iterations.shape[0]],
                   origin='lower')
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Construct full filepath
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                pad_inches=0, facecolor='black')
    print(f"Saved: {filepath}")
    plt.close()

def generate_variations():
    """Generate many Phoenix fractal variations"""
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    variations = [
        # Classic variations with different p values
        {
            'name': '01_classic_phoenix',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.56667, 'p_imag': 0.0,
            'colormap': 'fire'
        },
        {
            'name': '02_phoenix_negative_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': -0.5, 'p_imag': 0.0,
            'colormap': 'ice'
        },
        {
            'name': '03_phoenix_small_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.3, 'p_imag': 0.0,
            'colormap': 'emerald'
        },
        {
            'name': '04_phoenix_large_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.8, 'p_imag': 0.0,
            'colormap': 'sunset'
        },
        
        # Complex p variations
        {
            'name': '05_complex_p_spiral',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.5, 'p_imag': 0.1,
            'colormap': 'plasma'
        },
        {
            'name': '06_complex_p_twist',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.4, 'p_imag': 0.3,
            'colormap': 'ocean'
        },
        {
            'name': '07_complex_p_asymmetric',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.6, 'p_imag': -0.2,
            'colormap': 'volcano'
        },
        
        # Different c values
        {
            'name': '08_shifted_c',
            'c_real': -0.3, 'c_imag': 0.0,
            'p_real': 0.56667, 'p_imag': 0.0,
            'colormap': 'purple'
        },
        {
            'name': '09_complex_c',
            'c_real': -0.4, 'c_imag': 0.2,
            'p_real': 0.56667, 'p_imag': 0.0,
            'colormap': 'gold'
        },
        {
            'name': '10_zero_c',
            'c_real': 0.0, 'c_imag': 0.0,
            'p_real': 0.56667, 'p_imag': 0.0,
            'colormap': 'mint'
        },
        
        # Both c and p complex
        {
            'name': '11_both_complex_1',
            'c_real': -0.4, 'c_imag': 0.1,
            'p_real': 0.5, 'p_imag': 0.15,
            'colormap': 'fire'
        },
        {
            'name': '12_both_complex_2',
            'c_real': -0.3, 'c_imag': -0.15,
            'p_real': 0.45, 'p_imag': 0.2,
            'colormap': 'ice'
        },
        {
            'name': '12b_both_complex_2_green',
            'c_real': -0.3, 'c_imag': -0.15,
            'p_real': 0.45, 'p_imag': 0.2,
            'colormap': 'emerald'
        },
        
        # Extreme values
        {
            'name': '13_extreme_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 1.0, 'p_imag': 0.0,
            'colormap': 'emerald'
        },
        {
            'name': '14_negative_complex_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': -0.4, 'p_imag': -0.3,
            'colormap': 'sunset'
        },
        
        # Zoomed views of interesting regions
        {
            'name': '15_zoom_classic',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.56667, 'p_imag': 0.0,
            'zoom': 3.0, 'center_x': 0.3, 'center_y': 0.0,
            'max_iter': 512,
            'colormap': 'plasma'
        },
        {
            'name': '16_zoom_edge',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.56667, 'p_imag': 0.0,
            'zoom': 5.0, 'center_x': 0.5, 'center_y': 0.5,
            'max_iter': 512,
            'colormap': 'ocean'
        },
        
        # Symmetric variations
        {
            'name': '17_pure_imaginary_p',
            'c_real': -0.5, 'c_imag': 0.0,
            'p_real': 0.0, 'p_imag': 0.6,
            'colormap': 'volcano'
        },
        {
            'name': '18_pure_imaginary_c',
            'c_real': 0.0, 'c_imag': -0.5,
            'p_real': 0.56667, 'p_imag': 0.0,
            'colormap': 'purple'
        },
        
        # High detail variations
        {
            'name': '19_high_detail_1',
            'c_real': -0.45, 'c_imag': 0.0,
            'p_real': 0.52, 'p_imag': 0.08,
            'max_iter': 512,
            'colormap': 'gold'
        },
        {
            'name': '20_high_detail_2',
            'c_real': -0.55, 'c_imag': 0.0,
            'p_real': 0.6, 'p_imag': -0.1,
            'max_iter': 512,
            'colormap': 'mint'
        },
    ]
    
    print(f"\n{'='*60}")
    print(f"Generating {len(variations)} Phoenix Fractal Variations")
    print(f"{'='*60}\n")
    
    skipped = 0
    generated = 0
    
    for i, var in enumerate(variations, 1):
        filename = f"{var['name']}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"\n[{i}/{len(variations)}] {var['name']} - SKIPPED (already exists)")
            skipped += 1
            continue
        
        print(f"\n[{i}/{len(variations)}] {var['name']}")
        
        iterations = phoenix_fractal(
            width=1920,
            height=1080,
            max_iter=var.get('max_iter', 256),
            c_real=var['c_real'],
            c_imag=var['c_imag'],
            p_real=var['p_real'],
            p_imag=var['p_imag'],
            zoom=var.get('zoom', 1.0),
            center_x=var.get('center_x', 0.0),
            center_y=var.get('center_y', 0.0)
        )
        
        plot_fractal(
            iterations,
            filename,
            colormap=var.get('colormap', 'fire'),
            output_dir=output_dir
        )
        generated += 1
    
    print(f"\n{'='*60}")
    print(f"Generated: {generated}, Skipped: {skipped}, Total: {len(variations)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    start_time = time.time()
    generate_variations()
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")