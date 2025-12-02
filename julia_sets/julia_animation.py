import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

@jit(nopython=True, parallel=True, fastmath=True)
def julia_set_vectorized(height, width, c_real, c_imag, xmin, xmax, ymin, ymax, max_iter):
    """Vectorized Julia set calculation with Numba JIT compilation"""
    image = np.zeros((height, width), dtype=np.float32)
    
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    
    for i in prange(height):
        y = ymin + i * dy
        for j in range(width):
            x = xmin + j * dx
            
            # Initialize z
            zr = x
            zi = y
            
            # Iterate
            for n in range(max_iter):
                if zr*zr + zi*zi > 4.0:
                    image[i, j] = n
                    break
                
                # z = z^2 + c
                zr_new = zr*zr - zi*zi + c_real
                zi = 2*zr*zi + c_imag
                zr = zr_new
            else:
                image[i, j] = max_iter
    
    return image

def smooth_step(t):
    """Smoothstep interpolation for smoother transitions"""
    return t * t * (3 - 2 * t)

def ease_in_out(t):
    """Ease-in-out cubic interpolation"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def generate_frame_data(args):
    """Worker function for parallel frame generation"""
    frame, c_real, c_imag, width, height, xmin, xmax, ymin, ymax, max_iter, cmap_name = args
    
    # Generate Julia set data
    data = julia_set_vectorized(height, width, c_real, c_imag, xmin, xmax, ymin, ymax, max_iter)
    
    # Apply logarithmic scaling
    data_smooth = np.log(data + 1, dtype=np.float32)
    
    # Normalize to 0-255 range
    data_norm = ((data_smooth - data_smooth.min()) / (data_smooth.max() - data_smooth.min()) * 255).astype(np.uint8)
    
    # Apply colormap directly without matplotlib
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(data_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
    
    return frame, colored

def create_julia_animation(num_frames=180, width=1000, height=1000):
    """Create morphing animation of Julia set with parallel processing"""
    print("Generating Julia set animation with parallel processing...")
    
    radius = 0.7885
    
    # Pre-calculate all parameters
    frame_params = []
    for frame in range(num_frames):
        angle = 2 * np.pi * frame / num_frames
        c_real = radius * np.cos(angle)
        c_imag = radius * np.sin(angle)
        frame_params.append((frame, c_real, c_imag, width, height, -2, 2, -2, 2, 512, 'twilight_shifted'))
    
    # Generate frames in parallel
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(generate_frame_data, frame_params))
    
    # Sort by frame number and create PIL images
    results.sort(key=lambda x: x[0])
    frames = [Image.fromarray(result[1]) for result in results]
    
    print(f"All {num_frames} frames generated")
    
    # Save as animated GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), f"julia_animation_{timestamp}.gif")
    
    print("Saving GIF...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=False  # Faster saving
    )
    
    print(f"\nAnimation saved as: {output_path}")
    return output_path

def create_julia_zoom_animation(num_frames=180, width=1000, height=1000):
    """Create zooming animation with parallel processing"""
    print("Generating Julia set zoom animation with parallel processing...")
    
    c_real = -0.7269
    c_imag = 0.1889
    center_x, center_y = 0.0, 0.0
    zoom_start = 4.0
    zoom_end = 0.005
    
    # Pre-calculate all parameters
    frame_params = []
    for frame in range(num_frames):
        progress = frame / num_frames
        eased_progress = ease_in_out(progress)
        zoom = zoom_start * (zoom_end / zoom_start) ** eased_progress
        
        xmin = center_x - zoom
        xmax = center_x + zoom
        ymin = center_y - zoom
        ymax = center_y + zoom
        
        max_iter = int(512 + 500 * eased_progress)
        
        frame_params.append((frame, c_real, c_imag, width, height, xmin, xmax, ymin, ymax, max_iter, 'hot'))
    
    # Generate frames in parallel
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(generate_frame_data, frame_params))
    
    # Sort and create images
    results.sort(key=lambda x: x[0])
    frames = [Image.fromarray(result[1]) for result in results]
    
    print(f"All {num_frames} frames generated")
    
    # Save as animated GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), f"julia_zoom_{timestamp}.gif")
    
    print("Saving GIF...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=False
    )
    
    print(f"\nZoom animation saved as: {output_path}")
    return output_path

def create_julia_parameter_sweep(num_frames=240, width=1000, height=1000):
    """Create parameter sweep animation with parallel processing"""
    print("Generating Julia set parameter sweep animation with parallel processing...")
    
    interesting_points = [
        (-0.4, 0.6),
        (-0.7269, 0.1889),
        (0.285, 0.01),
        (-0.835, -0.2321),
        (-0.8, 0.156),
        (-0.4, -0.59),
        (-0.70176, -0.3842),
        (0.355, 0.355),
    ]
    
    colormaps = ['twilight_shifted', 'viridis', 'plasma', 'inferno', 'magma']
    
    # Pre-calculate all parameters
    frame_params = []
    for frame in range(num_frames):
        progress = frame / num_frames * len(interesting_points)
        segment_idx = int(progress) % len(interesting_points)
        next_idx = (segment_idx + 1) % len(interesting_points)
        
        t = progress - int(progress)
        t_smooth = smooth_step(t)
        
        p1 = interesting_points[segment_idx]
        p2 = interesting_points[next_idx]
        
        c_real = p1[0] * (1 - t_smooth) + p2[0] * t_smooth
        c_imag = p1[1] * (1 - t_smooth) + p2[1] * t_smooth
        
        cmap_idx = segment_idx % len(colormaps)
        cmap_name = colormaps[cmap_idx]
        
        frame_params.append((frame, c_real, c_imag, width, height, -2, 2, -2, 2, 512, cmap_name))
    
    # Generate frames in parallel
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(generate_frame_data, frame_params))
    
    # Sort and create images
    results.sort(key=lambda x: x[0])
    frames = [Image.fromarray(result[1]) for result in results]
    
    print(f"All {num_frames} frames generated")
    
    # Save as animated GIF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), f"julia_sweep_{timestamp}.gif")
    
    print("Saving GIF...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=False
    )
    
    print(f"\nParameter sweep animation saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    # Choose which animation to generate:
    # create_julia_animation(num_frames=180, width=1000, height=1000)
    # create_julia_zoom_animation(num_frames=180, width=1000, height=1000)
    create_julia_parameter_sweep(num_frames=240, width=1000, height=1000)