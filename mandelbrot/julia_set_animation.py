import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def julia_set(z, c, max_iter):
    """Calculate Julia set for complex number z with constant c"""
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_julia_frame(width, height, c_real, c_imag, xmin=-2, xmax=2, ymin=-2, ymax=2, max_iter=256):
    """Generate a single frame of the Julia set"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    c = complex(c_real, c_imag)
    image = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            z = complex(x[j], y[i])
            image[i, j] = julia_set(z, c, max_iter)
    
    return image

def create_julia_animation(num_frames=120, width=800, height=800):
    """Create morphing animation of Julia set"""
    frames = []
    
    print("Generating Julia set animation...")
    
    # Animate through a circular path in the complex plane
    # These values create interesting Julia sets
    radius = 0.7885
    
    for frame in range(num_frames):
        # Move in a circle around the origin
        angle = 2 * np.pi * frame / num_frames
        c_real = radius * np.cos(angle)
        c_imag = radius * np.sin(angle)
        
        # Generate Julia set data
        data = generate_julia_frame(width, height, c_real, c_imag)
        
        # Create colorful image
        fig, ax = plt.subplots(figsize=(10, 10), dpi=80, facecolor='black')
        ax.axis('off')
        ax.set_facecolor('black')
        
        # Use a vibrant colormap
        cmap = plt.cm.twilight_shifted
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear', extent=[-2, 2, -2, 2])
        
        plt.tight_layout(pad=0)
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Remove alpha channel
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
        
        print(f"Frame {frame + 1}/{num_frames} complete")
    
    # Save as animated GIF
    output_path = os.path.join(os.path.dirname(__file__), "julia_animation.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # milliseconds per frame
        loop=0  # infinite loop
    )
    
    print(f"\nAnimation saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    create_julia_animation(num_frames=120, width=800, height=800)
