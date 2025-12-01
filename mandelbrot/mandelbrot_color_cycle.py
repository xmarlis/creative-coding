import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import os

def mandelbrot(c, max_iter):
    """Calculate Mandelbrot set for complex number c"""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_mandelbrot_data(width, height, xmin, xmax, ymin, ymax, max_iter=256):
    """Generate Mandelbrot set data (only once)"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    image = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            image[i, j] = mandelbrot(c, max_iter)
    
    return image

def create_custom_colormap(shift):
    """Create a colormap with a phase shift for animation"""
    colors = []
    n_colors = 256
    
    for i in range(n_colors):
        # Create rainbow-like colors with phase shift
        phase = (i / n_colors + shift) * 2 * np.pi
        r = (np.sin(phase) + 1) / 2
        g = (np.sin(phase + 2 * np.pi / 3) + 1) / 2
        b = (np.sin(phase + 4 * np.pi / 3) + 1) / 2
        colors.append([r, g, b])
    
    return LinearSegmentedColormap.from_list('custom', colors)

def create_color_cycle_animation(num_frames=60, width=800, height=600):
    """Create color cycling animation of Mandelbrot set"""
    # Interesting zoom level
    target_x = -0.5
    target_y = 0.0
    zoom = 1.5
    
    xmin = target_x - zoom
    xmax = target_x + zoom
    ymin = target_y - zoom * (height / width)
    ymax = target_y + zoom * (height / width)
    
    print("Generating Mandelbrot data (one time)...")
    data = generate_mandelbrot_data(width, height, xmin, xmax, ymin, ymax, max_iter=256)
    
    frames = []
    print("Creating color-cycled frames...")
    
    for frame in range(num_frames):
        # Calculate color shift
        shift = frame / num_frames
        cmap = create_custom_colormap(shift)
        
        # Create colorful image
        fig, ax = plt.subplots(figsize=(10, 7.5), dpi=80, facecolor='black')
        ax.axis('off')
        ax.set_facecolor('black')
        
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear', extent=[xmin, xmax, ymin, ymax])
        
        plt.tight_layout(pad=0)
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
        
        print(f"Frame {frame + 1}/{num_frames} complete")
    
    # Save as animated GIF
    output_path = os.path.join(os.path.dirname(__file__), "mandelbrot_color_cycle.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    
    print(f"\nAnimation saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    create_color_cycle_animation(num_frames=60, width=800, height=600)
