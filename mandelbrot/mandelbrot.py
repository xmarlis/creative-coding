import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime

def mandelbrot(c, max_iter):
    """Calculate the Mandelbrot set for a complex number c"""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Generate the complete Mandelbrot set for given bounds"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y
    
    mandelbrot_set = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            mandelbrot_set[i, j] = mandelbrot(C[i, j], max_iter)
    
    return mandelbrot_set

def create_psychedelic_colormap(name, colors):
    """Create a psychedelic colormap from given colors"""
    return LinearSegmentedColormap.from_list(name, colors, N=256)

def generate_mandelbrot_image(params, colormap, filename, output_dir):
    """Generate and save a Mandelbrot image with given parameters"""
    mandel = mandelbrot_set(**params)
    
    plt.figure(figsize=(12, 12), dpi=150)
    plt.imshow(mandel, extent=[params['xmin'], params['xmax'], params['ymin'], params['ymax']], 
               cmap=colormap, origin='lower', interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def main():
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = r"c:\Users\Marlis\OneDrive\Dokumente\Projekte\creative-coding\mandelbrot"
    output_dir = os.path.join(base_dir, f"output_{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Output directory: {output_dir}\n")
    
    # Define psychedelic color schemes
    colormaps = {
        'acid_dream': create_psychedelic_colormap('acid_dream', 
            ['#FF00FF', '#00FFFF', '#FFFF00', '#FF0080', '#80FF00', '#0080FF']),
        'forest_trip': create_psychedelic_colormap('forest_trip',
            ['#000000', '#006400', '#228B22', '#32CD32', '#7FFF00', '#ADFF2F']),
        'rainbow_melt': create_psychedelic_colormap('rainbow_melt',
            ['#8A2BE2', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']),
        'electric_purple': create_psychedelic_colormap('electric_purple',
            ['#000000', '#4B0082', '#8A2BE2', '#DA70D6', '#FF00FF', '#FFFFFF']),
        'toxic_green': create_psychedelic_colormap('toxic_green',
            ['#000000', '#32CD32', '#ADFF2F', '#00FF00', '#7FFF00', '#FFFF00']),
        'cyber_pink': create_psychedelic_colormap('cyber_pink',
            ['#000080', '#8A2BE2', '#FF1493', '#FF69B4', '#FFB6C1', '#FFFFFF'])
    }
    
    # Define interesting parameter sets
    param_sets = [
        # Classic full view
        {'xmin': -2.5, 'xmax': 1.5, 'ymin': -2.0, 'ymax': 2.0, 'width': 800, 'height': 800, 'max_iter': 100},
        
        # Zoomed into interesting areas
        {'xmin': -0.75, 'xmax': -0.73, 'ymin': 0.1, 'ymax': 0.12, 'width': 800, 'height': 800, 'max_iter': 200},
        
        # Seahorse valley
        {'xmin': -0.743, 'xmax': -0.740, 'ymin': 0.126, 'ymax': 0.129, 'width': 800, 'height': 800, 'max_iter': 300},
        
        # Spiral region
        {'xmin': -0.8, 'xmax': -0.7, 'ymin': 0.0, 'ymax': 0.1, 'width': 800, 'height': 800, 'max_iter': 150},
        
        # Lightning pattern
        {'xmin': -1.25, 'xmax': -1.15, 'ymin': 0.0, 'ymax': 0.1, 'width': 800, 'height': 800, 'max_iter': 180},
        
        # Fractal tendrils
        {'xmin': -0.16, 'xmax': -0.14, 'ymin': 1.025, 'ymax': 1.045, 'width': 800, 'height': 800, 'max_iter': 250},
        
        # Mini mandelbrot
        {'xmin': -1.408, 'xmax': -1.398, 'ymin': -0.005, 'ymax': 0.005, 'width': 800, 'height': 800, 'max_iter': 400},
        
        # Feather pattern
        {'xmin': -0.235125, 'xmax': -0.235075, 'ymin': 0.827, 'ymax': 0.827075, 'width': 800, 'height': 800, 'max_iter': 500}
    ]
    
    # Generate images for each combination
    image_count = 0
    for i, (colormap_name, colormap) in enumerate(colormaps.items()):
        for j, params in enumerate(param_sets):
            filename = f"mandelbrot_{colormap_name}_variation_{j+1:02d}.png"
            generate_mandelbrot_image(params, colormap, filename, output_dir)
            image_count += 1
    
    print(f"\nGenerated {image_count} psychedelic Mandelbrot images!")
    print(f"Images saved to: {output_dir}")
    
    # Create a summary image showing thumbnails
    create_summary_grid(output_dir)

def create_summary_grid(output_dir):
    """Create a grid showing thumbnails of all generated images"""
    import glob
    from PIL import Image
    
    # Get all PNG files
    png_files = glob.glob(os.path.join(output_dir, "*.png"))
    if not png_files:
        return
    
    # Create a grid layout
    cols = 6
    rows = len(png_files) // cols + (1 if len(png_files) % cols else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, png_file in enumerate(png_files):
        row = i // cols
        col = i % cols
        
        try:
            img = plt.imread(png_file)
            axes[row, col].imshow(img)
            axes[row, col].set_title(os.path.basename(png_file), fontsize=8)
            axes[row, col].axis('off')
        except:
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(png_files), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "summary_grid.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary grid saved: summary_grid.png")

if __name__ == "__main__":
    main()
