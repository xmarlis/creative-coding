import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

# Try numba
try:
    from numba import njit, prange
    USE_NUMBA = True
    print("✓ Numba available")
except:
    USE_NUMBA = False
    print("⚠ No numba - will be slower")

# =============== SETTINGS ===============
IMG_SIZE = 1600
OUTPUT_DIR = Path(__file__).parent / datetime.now().strftime("phoenix_%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output: {OUTPUT_DIR}\n")

# =============== NUMBA VERSION ===============
if USE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def compute_phoenix(width, height, xmin, xmax, ymin, ymax, 
                       c_real, c_imag, p_real, p_imag, max_iter):
        result = np.zeros((height, width), dtype=np.float64)
        
        for row in prange(height):
            y = ymin + (ymax - ymin) * row / (height - 1)
            
            for col in range(width):
                x = xmin + (xmax - xmin) * col / (width - 1)
                
                # Start iteration
                z_real = x
                z_imag = y
                zprev_real = 0.0
                zprev_imag = 0.0
                
                iter_count = 0
                for i in range(max_iter):
                    # z^2
                    z_real_sq = z_real * z_real
                    z_imag_sq = z_imag * z_imag
                    
                    # Check escape
                    if z_real_sq + z_imag_sq > 1000.0:
                        iter_count = i
                        break
                    
                    # p * z_prev
                    p_zprev_real = p_real * zprev_real - p_imag * zprev_imag
                    p_zprev_imag = p_real * zprev_imag + p_imag * zprev_real
                    
                    # z_new = z^2 + c + p*z_prev
                    z_new_real = z_real_sq - z_imag_sq + c_real + p_zprev_real
                    z_new_imag = 2.0 * z_real * z_imag + c_imag + p_zprev_imag
                    
                    # Update
                    zprev_real = z_real
                    zprev_imag = z_imag
                    z_real = z_new_real
                    z_imag = z_new_imag
                else:
                    iter_count = max_iter
                
                result[row, col] = iter_count
        
        return result

# =============== NUMPY VERSION ===============
def compute_phoenix_numpy(width, height, xmin, xmax, ymin, ymax,
                         c_real, c_imag, p_real, p_imag, max_iter):
    print("  Using NumPy (slower)...")
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    
    Z = X + 1j * Y
    Z_prev = np.zeros_like(Z)
    C = complex(c_real, c_imag)
    P = complex(p_real, p_imag)
    
    result = np.zeros(Z.shape, dtype=np.float64)
    mask = np.ones(Z.shape, dtype=bool)
    
    for i in range(max_iter):
        # Compute new Z
        Z_new = Z*Z + C + P*Z_prev
        
        # Check escape
        escaped = mask & (np.abs(Z_new) > 1000)
        result[escaped] = i
        mask[escaped] = False
        
        if not mask.any():
            break
        
        # Update
        Z_prev = Z.copy()
        Z = Z_new
    
    result[mask] = max_iter
    return result

# =============== MAIN GENERATOR ===============
def generate_phoenix(name, xmin, xmax, ymin, ymax, c_real, c_imag, 
                    p_real, p_imag, max_iter, cmap, gamma):
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"  Region: [{xmin:.2f}, {xmax:.2f}] x [{ymin:.2f}, {ymax:.2f}]")
    print(f"  c = {c_real:.4f} + {c_imag:.4f}i")
    print(f"  p = {p_real:.4f} + {p_imag:.4f}i")
    print(f"  max_iter = {max_iter}")
    
    t0 = time.time()
    
    if USE_NUMBA:
        data = compute_phoenix(IMG_SIZE, IMG_SIZE, xmin, xmax, ymin, ymax,
                              c_real, c_imag, p_real, p_imag, max_iter)
    else:
        data = compute_phoenix_numpy(IMG_SIZE, IMG_SIZE, xmin, xmax, ymin, ymax,
                                    c_real, c_imag, p_real, p_imag, max_iter)
    
    compute_time = time.time() - t0
    
    # Check data
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    unique = len(np.unique(data))
    
    print(f"  Computed in {compute_time:.2f}s")
    print(f"  Stats: min={min_val:.1f}, max={max_val:.1f}, mean={mean_val:.1f}")
    print(f"  Unique values: {unique:,}")
    
    # Normalize
    if max_val > min_val:
        normalized = (data - min_val) / (max_val - min_val)
    else:
        print("  WARNING: No variation in data!")
        normalized = np.zeros_like(data)
    
    # Apply gamma
    img = np.power(normalized, gamma)
    
    # Render
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(img, cmap=cmap, origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save
    filepath = OUTPUT_DIR / f"{name}.png"
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    
    print(f"  Saved: {filepath.name}")
    print(f"  Total time: {time.time() - t0:.2f}s")

# =============== VARIANTS WITH TESTED PARAMETERS ===============

def main():
    start_time = time.time()
    
    # Variant 1: Classic Phoenix Set (the most iconic)
    generate_phoenix(
        name="1_classic_phoenix",
        xmin=-2.0, xmax=2.0,
        ymin=-2.0, ymax=2.0,
        c_real=0.5667, c_imag=-0.5,
        p_real=-0.5, p_imag=0.0,
        max_iter=500,
        cmap='hot',
        gamma=0.6
    )
    
    # Variant 2: Phoenix Wings
    generate_phoenix(
        name="2_phoenix_wings", 
        xmin=-1.5, xmax=1.5,
        ymin=-1.5, ymax=1.5,
        c_real=0.5, c_imag=-0.5,
        p_real=-0.5, p_imag=0.0,
        max_iter=500,
        cmap='magma',
        gamma=0.7
    )
    
    # Variant 3: Julia-like variation
    generate_phoenix(
        name="3_electric_phoenix",
        xmin=-1.2, xmax=1.2,
        ymin=-1.2, ymax=1.2,
        c_real=0.55, c_imag=-0.48,
        p_real=-0.52, p_imag=0.0,
        max_iter=600,
        cmap='plasma',
        gamma=0.5
    )
    
    # Variant 4: Zoom into interesting region
    generate_phoenix(
        name="4_phoenix_detail",
        xmin=-0.5, xmax=0.5,
        ymin=-0.5, ymax=0.5,
        c_real=0.5667, c_imag=-0.5,
        p_real=-0.5, p_imag=0.0,
        max_iter=800,
        cmap='inferno',
        gamma=0.8
    )
    
    # Variant 5: Wide view with different parameters
    generate_phoenix(
        name="5_cosmic_bird",
        xmin=-2.5, xmax=2.5,
        ymin=-2.5, ymax=2.5,
        c_real=0.56667, c_imag=-0.5,
        p_real=-0.5, p_imag=0.0,
        max_iter=700,
        cmap='twilight',
        gamma=0.9
    )
    
    # Variant 6: Variation with imaginary p
    generate_phoenix(
        name="6_twisted_phoenix",
        xmin=-1.8, xmax=1.8,
        ymin=-1.8, ymax=1.8,
        c_real=0.52, c_imag=-0.51,
        p_real=-0.48, p_imag=0.03,
        max_iter=600,
        cmap='viridis',
        gamma=0.65
    )
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average: {total_time/6:.1f}s per image")
    print(f"Location: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()