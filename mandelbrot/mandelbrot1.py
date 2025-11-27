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

def generate_mandelbrot_frame(width, height, xmin, xmax, ymin, ymax, max_iter=256):
    """Generate a single frame of the Mandelbrot set"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    image = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            image[i, j] = mandelbrot(c, max_iter)
    
    return image

def create_mandelbrot_animation(num_frames=60, width=800, height=600):
    """Create zooming animation of Mandelbrot set"""
    # Zoom target: interesting point near the edge
    target_x = -0.7453
    target_y = 0.1128
    
    # Initial view
    initial_width = 3.5
    initial_height = 2.5
    
    # Final zoom level
    final_zoom = 0.00001
    
    frames = []
    
    print("Generating frames...")
    for frame in range(num_frames):
        # Exponential zoom
        progress = frame / (num_frames - 1)
        zoom = initial_width * (final_zoom / initial_width) ** progress
        
        xmin = target_x - zoom
        xmax = target_x + zoom
        ymin = target_y - zoom * (height / width)
        ymax = target_y + zoom * (height / width)
        
        # Generate Mandelbrot data
        data = generate_mandelbrot_frame(width, height, xmin, xmax, ymin, ymax)
        
        # Create colorful image
        fig, ax = plt.subplots(figsize=(10, 7.5), dpi=80, facecolor='black')
        ax.axis('off')
        ax.set_facecolor('black')
        
        # Use a nice colormap
        cmap = plt.cm.Greens_r  # _r reverses the colormap
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear', extent=[xmin, xmax, ymin, ymax])
        
        plt.tight_layout(pad=0)
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Remove alpha channel, keep RGB
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
        
        print(f"Frame {frame + 1}/{num_frames} complete")
    
    # Save as animated GIF
    output_path = os.path.join(os.path.dirname(__file__), "mandelbrot_animation.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame (increased from 100 to 200 for slower animation)
        loop=0  # infinite loop
    )
    
    print(f"\nAnimation saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    create_mandelbrot_animation(num_frames=60, width=800, height=600)
