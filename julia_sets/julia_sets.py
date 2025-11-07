import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime

def compute_julia_set(h, w, c, max_iter=100):
    """
    Compute the Julia set for a given complex parameter c.
    
    Parameters:
    h, w (int): Height and width of the output image
    c (complex): The complex parameter that defines this particular Julia set
    max_iter (int): Maximum number of iterations
    
    Returns:
    numpy.ndarray: 2D array containing the iteration count for each point
    """
    # Define the region in the complex plane
    y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
    z = x + y*1j
    
    # Initialize the iteration count array
    iter_counts = max_iter + np.zeros(z.shape, dtype=int)
    
    # Create a mask for points that haven't escaped yet
    mask = np.ones(z.shape, dtype=bool)
    
    # Iterate z_n+1 = z_n^2 + c and track iterations until escape
    for i in range(max_iter):
        z[mask] = z[mask]**2 + c
        # The modulus squared is easier to compute than the modulus
        escaped = np.abs(z) > 2.0
        # Update the iteration counts for points that escape on this iteration
        iter_counts[mask & escaped] = i
        # Update the mask to exclude points that have escaped
        mask &= ~escaped
    
    return iter_counts

def create_colormap(name="classic"):
    """Create a custom colormap for the Julia set visualization"""
    if name == "classic":
        colors = [(0, 0, 0), (0, 0, 0.5), (0, 0, 1), (0, 0.5, 1),
                  (0, 1, 1), (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]
    elif name == "fire":
        colors = [(0, 0, 0), (0.2, 0, 0), (0.5, 0, 0), (1, 0, 0), 
                  (1, 0.5, 0), (1, 0.8, 0), (1, 1, 0.2)]
    elif name == "ocean":
        colors = [(0, 0, 0), (0, 0.1, 0.2), (0, 0.2, 0.4), (0, 0.3, 0.6),
                  (0, 0.5, 0.8), (0, 0.7, 0.9), (0.1, 0.9, 1.0)]
    elif name == "forest":
        colors = [(0, 0, 0), (0, 0.1, 0), (0, 0.2, 0), (0.1, 0.3, 0.1),
                  (0.2, 0.5, 0.2), (0.3, 0.7, 0.3), (0.6, 0.9, 0.6)]
    elif name == "psychedelic":
        colors = [(0, 0, 0), (0.7, 0, 0.7), (1, 0, 0), (1, 0.5, 0),
                  (1, 1, 0), (0, 1, 0), (0, 0, 1), (0.7, 0, 0.7)]
    elif name == "grayscale":
        colors = [(0, 0, 0), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), 
                  (0.6, 0.6, 0.6), (0.8, 0.8, 0.8), (1, 1, 1)]
    elif name == "sunset":
        colors = [(0, 0, 0.2), (0.2, 0, 0.4), (0.5, 0, 0.5), (0.7, 0.2, 0.4),
                  (0.9, 0.4, 0.2), (1, 0.6, 0.1), (1, 0.9, 0.3)]
    else:
        # Default to classic
        return create_colormap("classic")
        
    return LinearSegmentedColormap.from_list(f'julia_colormap_{name}', colors, N=256)

def get_available_colormaps():
    """Return a list of available colormap names"""
    return ["classic", "fire", "ocean", "forest", "psychedelic", "grayscale", "sunset"]

def get_unique_filename(base_filename, output_dir):
    """
    Generate a unique filename by adding a timestamp to prevent overwrites.
    
    Parameters:
    base_filename (str): The desired base filename
    output_dir (str): The directory where the file will be saved
    
    Returns:
    str: A unique filename with timestamp that doesn't exist in the output directory
    """
    base_name, ext = os.path.splitext(base_filename)
    
    # Always add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{base_name}_{timestamp}{ext}"
    
    # If timestamp version exists (unlikely but possible), add counter
    counter = 1
    while os.path.exists(os.path.join(output_dir, unique_filename)):
        unique_filename = f"{base_name}_{timestamp}_{counter}{ext}"
        counter += 1
    
    return unique_filename

def plot_julia_set(c, width=1000, height=1000, max_iter=100, filename=None, colormap_name="classic", show_title=True):
    """
    Generate and optionally save a Julia set image.
    
    Parameters:
    c (complex): The complex parameter for the Julia set
    width, height (int): Dimensions of the output image
    max_iter (int): Maximum number of iterations
    filename (str, optional): If provided, save the image to this filename
    colormap_name (str): Name of the colormap to use
    show_title (bool): Whether to show the title (default: True)
    """
    julia_data = compute_julia_set(height, width, c, max_iter)
    
    # Create a figure with tight layout
    plt.figure(figsize=(10, 10), frameon=False)
    plt.axis('off')
    
    # Plot the Julia set
    colormap = create_colormap(colormap_name)
    plt.imshow(julia_data, cmap=colormap, extent=[-1.5, 1.5, -1.5, 1.5])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if show_title:
        plt.title(f"Julia Set for c = {c:.4f}")
    
    if filename:
        output_dir = os.path.join(os.path.dirname(__file__), "julia_sets_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Add colormap name to filename
        base_name, ext = os.path.splitext(filename)
        filename_with_colormap = f"{base_name}_{colormap_name}{ext}"
        
        # Get unique filename to prevent overwrites
        unique_filename = get_unique_filename(filename_with_colormap, output_dir)
        
        filepath = os.path.join(output_dir, unique_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {filepath}")
    
    plt.show()

def explore_julia_sets_from_mandelbrot():
    """
    Generate Julia sets for various points related to the Mandelbrot set.
    """
    # Some interesting points related to the Mandelbrot set
    interesting_points = [
        (-0.7269, 0.1889),  # Dendrite-like structure
        (-0.8, 0.156),      # Close to the main cardioid
        (-0.75, 0.11),      # Spirals
        (-0.1, 0.8),        # Disconnected set
        (0.285, 0.01),      # Fractal pattern
        (-1.417022285618, 0.0)  # Period-doubling point
    ]
    
    # Get colormap choice
    print("Available colormaps:")
    colormaps = get_available_colormaps()
    
    for i, cmap in enumerate(colormaps, 1):
        print(f"{i}. {cmap}")
    
    cmap_choice = input("Enter colormap number or 'all' to generate all: ")
    
    if cmap_choice.lower() == 'all':
        selected_colormaps = colormaps
    else:
        try:
            idx = int(cmap_choice) - 1
            if 0 <= idx < len(colormaps):
                selected_colormaps = [colormaps[idx]]
            else:
                print("Invalid choice, using classic colormap.")
                selected_colormaps = ["classic"]
        except ValueError:
            print("Invalid input, using classic colormap.")
            selected_colormaps = ["classic"]
    
    # Ask if titles should be shown on saved images
    show_title = input("Include titles on saved images? (y/n): ").lower() == 'y'
    
    # Generate images
    for idx, (real, imag) in enumerate(interesting_points):
        c = complex(real, imag)
        base_filename = f"julia_set_{idx+1}.png"
        print(f"Generating Julia set for c = {c}")
        
        for cmap_name in selected_colormaps:
            plot_julia_set(c, filename=base_filename, colormap_name=cmap_name, show_title=show_title)

def interactive_exploration():
    """
    Allow the user to enter custom values for the Julia set parameter c.
    """
    print("\nJulia Set Explorer")
    print("------------------")
    print("Enter values for the complex parameter c = a + bi")
    
    # Print available colormaps
    print("Available colormaps:")
    colormaps = get_available_colormaps()
    for i, cmap in enumerate(colormaps, 1):
        print(f"{i}. {cmap}")
    
    cmap_idx = input("Select colormap number (default=1): ")
    try:
        idx = int(cmap_idx) - 1
        if 0 <= idx < len(colormaps):
            chosen_colormap = colormaps[idx]
        else:
            chosen_colormap = "classic"
    except ValueError:
        chosen_colormap = "classic"
    
    print(f"Using colormap: {chosen_colormap}")
    
    # Ask if titles should be shown
    show_title = input("Show titles on images? (y/n): ").lower() == 'y'
    
    while True:
        try:
            a = float(input("Enter real part a (or 'q' to quit): "))
            b = float(input("Enter imaginary part b: "))
            c = complex(a, b)
            
            # Ask if the user wants to save the image
            save_option = input("Save this image? (y/n): ")
            filename = None
            if save_option.lower() == 'y':
                filename = f"julia_custom_r{a:.6f}_i{b:.6f}.png"
            
            plot_julia_set(c, filename=filename, colormap_name=chosen_colormap, show_title=show_title)
        except ValueError as e:
            if 'q' in str(e).lower():
                print("Exiting...")
                break
            print(f"Error: {e}. Please enter valid numbers.")
        
        choice = input("\nGenerate another Julia set? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    print("Julia Sets Explorer")
    print("------------------")
    print("1. Generate a set of interesting Julia sets")
    print("2. Interactive exploration mode")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        explore_julia_sets_from_mandelbrot()
    elif choice == '2':
        interactive_exploration()
    else:
        print("Invalid choice. Exiting.")
