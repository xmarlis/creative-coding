import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os

def phoenix_fractal(width=1920, height=1080, max_iter=256, 
                   c_real=-0.5, c_imag=0.0, 
                   p_real=0.56667, p_imag=0.0,
                   zoom=1.0, center_x=0.0, center_y=0.0):
    """
    Generate a Phoenix fractal.
    
    The Phoenix fractal is defined by the iteration:
    z_{n+1} = z_n^2 + c + p * z_{n-1}
    
    Parameters:
    -----------
    width, height : int
        Image dimensions in pixels
    max_iter : int
        Maximum number of iterations
    c_real, c_imag : float
        Real and imaginary parts of parameter c
    p_real, p_imag : float
        Real and imaginary parts of parameter p
        Classic values: p = 0.56667 (or -0.5)
    zoom : float
        Zoom factor (higher = more zoomed in)
    center_x, center_y : float
        Center coordinates of the view
    """
    
    print(f"Generating Phoenix fractal...")
    print(f"Parameters: c = {c_real} + {c_imag}i, p = {p_real} + {p_imag}i")
    print(f"Resolution: {width}x{height}, Max iterations: {max_iter}")
    
    start_time = time.time()
    
    # Define the complex plane boundaries
    # Phoenix fractals are typically viewed in a smaller range than Mandelbrot
    x_min, x_max = center_x - 1.5/zoom, center_x + 1.5/zoom
    y_min, y_max = center_y - 1.5/zoom * height/width, center_y + 1.5/zoom * height/width
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays
    Z = X + 1j*Y  # Current z value
    Z_prev = np.zeros_like(Z)  # Previous z value (z_{n-1})
    iterations = np.zeros(Z.shape, dtype=int)
    
    # Complex parameters
    c = c_real + 1j*c_imag
    p = p_real + 1j*p_imag
    
    # Create mask for points that haven't escaped
    mask = np.ones(Z.shape, dtype=bool)
    
    # Iterate
    for i in range(max_iter):
        # Compute next iteration: z_{n+1} = z_n^2 + c + p * z_{n-1}
        Z_temp = Z**2 + c + p * Z_prev
        
        # Update Z_prev and Z for next iteration
        Z_prev = Z
        Z = Z_temp
        
        # Check which points have escaped (magnitude > bailout)
        escaped = np.abs(Z) > 10.0
        
        # Record iteration count for newly escaped points
        newly_escaped = escaped & mask
        iterations[newly_escaped] = i
        
        # Update mask - remove escaped points
        mask = mask & ~escaped
        
        if i % 50 == 0:
            escaped_count = np.sum(~mask)
            print(f"Iteration {i}/{max_iter} - {escaped_count} points escaped")
        
        # Early exit if all points have escaped
        if not mask.any():
            print(f"All points escaped by iteration {i}")
            break
    
    # Points that never escaped get max iterations
    iterations[mask] = max_iter
    
    elapsed = time.time() - start_time
    print(f"Computation completed in {elapsed:.2f} seconds")
    
    return iterations

def plot_fractal(iterations, filename='phoenix_fractal.png', dpi=150,
                colormap='custom', output_dir=None):
    """
    Plot and save the fractal.
    
    Parameters:
    -----------
    iterations : ndarray
        Array of iteration counts
    filename : str
        Output filename
    dpi : int
        Resolution for saving
    colormap : str
        Color scheme ('custom', 'hot', 'twilight', 'viridis', etc.)
    output_dir : str
        Directory to save the output file
    """
    
    print(f"Creating visualization...")
    print(f"Iteration range: {iterations.min()} to {iterations.max()}")
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='black')
    
    # Normalize iterations for better color distribution
    # Use square root for smoother gradients
    norm_iterations = np.sqrt(iterations.astype(float))
    
    if colormap == 'custom':
        # Create custom colormap (phoenix colors: orange, red, purple, black)
        colors = ['#000033', '#1a0066', '#4a0080', '#8000ff', '#ff0080', 
                  '#ff3300', '#ff6600', '#ffaa00', '#ffff00', '#ffffff']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('phoenix', colors, N=n_bins)
    else:
        cmap = plt.get_cmap(colormap)
    
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
    
    # Save the figure
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                pad_inches=0, facecolor='black')
    print(f"Saved fractal to: {filepath}")
    
    plt.close()

def main():
    """Generate various Phoenix fractal variations"""
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Classic Phoenix fractal
    print("\n=== Classic Phoenix Fractal ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=256,
        c_real=-0.5,
        c_imag=0.0,
        p_real=0.56667,
        p_imag=0.0,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_classic.png', colormap='custom', output_dir=output_dir)
    
    # Variation 1: Different p parameter
    print("\n=== Phoenix Variation 1 (p = -0.5) ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=256,
        c_real=-0.5,
        c_imag=0.0,
        p_real=-0.5,
        p_imag=0.0,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_variation1.png', colormap='twilight', output_dir=output_dir)
    
    # Variation 2: Complex p parameter
    print("\n=== Phoenix Variation 2 (complex p) ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=256,
        c_real=-0.4,
        c_imag=0.0,
        p_real=0.5,
        p_imag=0.1,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_variation2.png', colormap='hot', output_dir=output_dir)
    
    # Variation 3: Spiral Phoenix (complex c and p)
    print("\n=== Phoenix Spiral ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=300,
        c_real=-0.3,
        c_imag=0.5,
        p_real=0.4,
        p_imag=0.3,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_spiral.png', colormap='viridis', output_dir=output_dir)
    
    # Variation 4: Feather Phoenix (delicate structures)
    print("\n=== Phoenix Feather ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=400,
        c_real=-0.2,
        c_imag=0.65,
        p_real=0.3,
        p_imag=-0.4,
        zoom=1.2,
        center_x=0.1,
        center_y=0.1
    )
    plot_fractal(iterations, 'phoenix_feather.png', colormap='plasma', output_dir=output_dir)
    
    # Variation 5: Dragon Phoenix (asymmetric)
    print("\n=== Phoenix Dragon ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=350,
        c_real=-0.6,
        c_imag=0.2,
        p_real=0.6,
        p_imag=0.15,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_dragon.png', colormap='inferno', output_dir=output_dir)
    
    # Variation 6: Ice Phoenix (cool colors, symmetric)
    print("\n=== Phoenix Ice ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=300,
        c_real=-0.45,
        c_imag=0.0,
        p_real=0.52,
        p_imag=0.25,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_ice.png', colormap='cool', output_dir=output_dir)
    
    # Variation 7: Nebula Phoenix (high iteration, complex)
    print("\n=== Phoenix Nebula ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=500,
        c_real=-0.35,
        c_imag=0.35,
        p_real=0.45,
        p_imag=0.35,
        zoom=1.5,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_nebula.png', colormap='twilight_shifted', output_dir=output_dir)
    
    # Variation 8: Crystalline Phoenix (sharp features)
    print("\n=== Phoenix Crystal ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=400,
        c_real=-0.55,
        c_imag=0.1,
        p_real=0.58,
        p_imag=-0.05,
        zoom=1.0,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_crystal.png', colormap='spring', output_dir=output_dir)
    
    # Zoomed view - Classic detail
    print("\n=== Phoenix Zoomed View ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=512,
        c_real=-0.5,
        c_imag=0.0,
        p_real=0.56667,
        p_imag=0.0,
        zoom=3.0,
        center_x=0.3,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_zoomed.png', colormap='custom', output_dir=output_dir)
    
    # Variation 9: Deep zoom - Spiral detail
    print("\n=== Phoenix Deep Spiral Detail ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=600,
        c_real=-0.3,
        c_imag=0.5,
        p_real=0.4,
        p_imag=0.3,
        zoom=4.0,
        center_x=0.25,
        center_y=-0.15
    )
    plot_fractal(iterations, 'phoenix_spiral_detail.png', colormap='magma', output_dir=output_dir)
    
    # Variation 10: Butterfly Phoenix (symmetric beauty)
    print("\n=== Phoenix Butterfly ===")
    iterations = phoenix_fractal(
        width=1920,
        height=1080,
        max_iter=350,
        c_real=-0.25,
        c_imag=0.0,
        p_real=0.5,
        p_imag=0.5,
        zoom=1.3,
        center_x=0.0,
        center_y=0.0
    )
    plot_fractal(iterations, 'phoenix_butterfly.png', colormap='RdYlBu', output_dir=output_dir)
    
    print("\n=== All fractals generated successfully! ===")

if __name__ == "__main__":
    main()