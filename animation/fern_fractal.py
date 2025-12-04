#!/usr/bin/env python3
"""
Barnsley Fern Animation - IMPROVED PSYCHEDELIC COLORS
Enhanced color schemes with better contrast
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import os
import subprocess


def check_ffmpeg():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def smootherstep(t):
    """
    Ultra-smooth interpolation function
    f(t) = 6t^5 - 15t^4 + 10t^3
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_fern_fast(params, n_points=40000):
    """
    OPTIMIZED: Generate fern fractal faster with vectorized operations
    """
    # Pre-allocate arrays
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    # Starting point
    x[0], y[0] = 0, 0
    
    # Unpack parameters
    p1, p2, p3, p4 = params['probs']
    cumulative_probs = [p1, p1+p2, p1+p2+p3]
    
    # Generate all random numbers at once (MUCH faster!)
    random_values = np.random.random(n_points-1)
    
    # Pre-compute transformation matrices for faster lookup
    transforms = [
        (params['f1_a'], params['f1_b'], params['f1_c'], params['f1_d'], params['f1_e'], params['f1_f']),
        (params['f2_a'], params['f2_b'], params['f2_c'], params['f2_d'], params['f2_e'], params['f2_f']),
        (params['f3_a'], params['f3_b'], params['f3_c'], params['f3_d'], params['f3_e'], params['f3_f']),
        (params['f4_a'], params['f4_b'], params['f4_c'], params['f4_d'], params['f4_e'], params['f4_f']),
    ]
    
    # Generate points
    for i in range(1, n_points):
        r = random_values[i-1]
        
        # Select transformation based on probability
        if r < cumulative_probs[0]:
            t_idx = 0
        elif r < cumulative_probs[1]:
            t_idx = 1
        elif r < cumulative_probs[2]:
            t_idx = 2
        else:
            t_idx = 3
        
        a, b, c, d, e, f = transforms[t_idx]
        x[i] = a * x[i-1] + b * y[i-1] + e
        y[i] = c * x[i-1] + d * y[i-1] + f
    
    return x, y


def interpolate_params(params1, params2, t):
    """
    Smoothly interpolate between two parameter sets
    """
    result = {}
    for key in params1.keys():
        if isinstance(params1[key], (list, tuple)):
            result[key] = [
                params1[key][i] * (1 - t) + params2[key][i] * t
                for i in range(len(params1[key]))
            ]
        else:
            result[key] = params1[key] * (1 - t) + params2[key] * t
    return result


def create_fern_animation(filename='fern_psychedelic_improved.mp4', 
                         frames=360,
                         n_points=40000,
                         bins=500,
                         fps=30):
    """
    Create psychedelic fern animation with IMPROVED COLORS
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    print(f'Output will be saved to: {output_path}')
    
    # Different fern parameter sets for morphing
    fern_params = [
        # Classic Barnsley Fern
        {
            'probs': [0.01, 0.85, 0.07, 0.07],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.85, 'f2_b': 0.04, 'f2_c': -0.04, 'f2_d': 0.85, 'f2_e': 0, 'f2_f': 1.6,
            'f3_a': 0.2, 'f3_b': -0.26, 'f3_c': 0.23, 'f3_d': 0.22, 'f3_e': 0, 'f3_f': 1.6,
            'f4_a': -0.15, 'f4_b': 0.28, 'f4_c': 0.26, 'f4_d': 0.24, 'f4_e': 0, 'f4_f': 0.44,
        },
        # Tall and sparse
        {
            'probs': [0.01, 0.88, 0.055, 0.055],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.87, 'f2_b': 0.02, 'f2_c': -0.02, 'f2_d': 0.87, 'f2_e': 0, 'f2_f': 1.5,
            'f3_a': 0.18, 'f3_b': -0.28, 'f3_c': 0.25, 'f3_d': 0.20, 'f3_e': 0, 'f3_f': 1.7,
            'f4_a': -0.18, 'f4_b': 0.28, 'f4_c': 0.25, 'f4_d': 0.20, 'f4_e': 0, 'f4_f': 0.5,
        },
        # Bushy fern
        {
            'probs': [0.01, 0.82, 0.085, 0.085],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.82, 'f2_b': 0.06, 'f2_c': -0.06, 'f2_d': 0.83, 'f2_e': 0, 'f2_f': 1.6,
            'f3_a': 0.25, 'f3_b': -0.22, 'f3_c': 0.22, 'f3_d': 0.25, 'f3_e': 0, 'f3_f': 1.5,
            'f4_a': -0.22, 'f4_b': 0.25, 'f4_c': 0.24, 'f4_d': 0.26, 'f4_e': 0, 'f4_f': 0.5,
        },
        # Cycad-like (palm)
        {
            'probs': [0.01, 0.84, 0.075, 0.075],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.88, 'f2_b': 0.0, 'f2_c': 0.0, 'f2_d': 0.88, 'f2_e': 0, 'f2_f': 1.4,
            'f3_a': 0.15, 'f3_b': -0.35, 'f3_c': 0.28, 'f3_d': 0.18, 'f3_e': 0, 'f3_f': 1.8,
            'f4_a': -0.15, 'f4_b': 0.35, 'f4_c': 0.28, 'f4_d': 0.18, 'f4_e': 0, 'f4_f': 0.3,
        },
        # Twisted fern
        {
            'probs': [0.01, 0.86, 0.065, 0.065],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.84, 'f2_b': 0.08, 'f2_c': -0.08, 'f2_d': 0.86, 'f2_e': 0.1, 'f2_f': 1.55,
            'f3_a': 0.22, 'f3_b': -0.24, 'f3_c': 0.26, 'f3_d': 0.20, 'f3_e': -0.1, 'f3_f': 1.65,
            'f4_a': -0.16, 'f4_b': 0.30, 'f4_c': 0.28, 'f4_d': 0.22, 'f4_e': 0.1, 'f4_f': 0.4,
        },
        # Wide spreading fern
        {
            'probs': [0.01, 0.80, 0.095, 0.095],
            'f1_a': 0, 'f1_b': 0, 'f1_c': 0, 'f1_d': 0.16, 'f1_e': 0, 'f1_f': 0,
            'f2_a': 0.80, 'f2_b': 0.05, 'f2_c': -0.05, 'f2_d': 0.81, 'f2_e': 0, 'f2_f': 1.7,
            'f3_a': 0.28, 'f3_b': -0.20, 'f3_c': 0.20, 'f3_d': 0.28, 'f3_e': 0, 'f3_f': 1.4,
            'f4_a': -0.20, 'f4_b': 0.28, 'f4_c': 0.22, 'f4_d': 0.30, 'f4_e': 0, 'f4_f': 0.6,
        },
    ]
    
    # Add first params at end for seamless looping
    fern_params.append(fern_params[0])
    
    # IMPROVED PSYCHEDELIC COLORMAPS - Multiple options with better contrast
    colormaps = [
        # 1. Electric Neon Rainbow
        ['#000000', '#FF0099', '#FF00FF', '#9900FF', '#0099FF', '#00FFFF', 
         '#00FF99', '#99FF00', '#FFFF00', '#FF9900', '#FF0000', '#FFFFFF'],
        
        # 2. Sunset Fire
        ['#000000', '#1a0033', '#4d0099', '#8000FF', '#FF00FF', '#FF0080',
         '#FF0040', '#FF3300', '#FF6600', '#FF9900', '#FFCC00', '#FFFF99'],
        
        # 3. Ocean Aurora
        ['#000000', '#001a33', '#003366', '#0066CC', '#0099FF', '#00CCFF',
         '#00FFCC', '#00FF99', '#66FF66', '#CCFF00', '#FFFF00', '#FFFFFF'],
        
        # 4. Toxic Acid
        ['#000000', '#003300', '#006600', '#009900', '#00CC00', '#00FF00',
         '#66FF00', '#99FF00', '#CCFF00', '#FFFF00', '#FFCC00', '#FF9900'],
        
        # 5. Deep Space
        ['#000000', '#0d0033', '#1a0066', '#330099', '#6600CC', '#9900FF',
         '#CC00FF', '#FF00CC', '#FF0099', '#FF3399', '#FF66CC', '#FFCCFF'],
        
        # 6. Candy Dreams
        ['#000000', '#330033', '#660066', '#990099', '#CC00CC', '#FF00FF',
         '#FF33CC', '#FF66CC', '#FF99CC', '#FFCCCC', '#FFCCFF', '#FFFFFF'],
    ]
    
    # Create colormaps
    cmaps = [LinearSegmentedColormap.from_list(f'psychedelic_{i}', colors, N=256) 
             for i, colors in enumerate(colormaps)]
    
    # Set up the figure with BLACK background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Initialize with first fern
    x, y = generate_fern_fast(fern_params[0], n_points)
    
    # Create 2D histogram for smooth coloring
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, 
                                       range=[[-3, 3], [0, 11]])
    
    # Apply logarithmic scaling for better visualization
    H = np.log(H + 1)
    
    # Start with first colormap
    im = ax.imshow(H.T, extent=[-3, 3, 0, 11], 
                   cmap=cmaps[0], interpolation='bilinear', origin='lower',
                   aspect='auto')
    
    def update(frame):
        """Update function for animation with continuous smooth transitions"""
        # Calculate which params we're transitioning between
        total_transitions = len(fern_params) - 1
        frames_per_transition = frames / total_transitions
        
        # Current transition index
        transition_idx = min(int(frame / frames_per_transition), total_transitions - 1)
        next_transition_idx = transition_idx + 1
        
        # Interpolation factor (0 to 1) within current transition
        t = (frame - transition_idx * frames_per_transition) / frames_per_transition
        
        # Apply smootherstep for ultra-smooth interpolation
        t_smooth = smootherstep(t)
        
        # Smoothly interpolate between parameter sets
        current_params = interpolate_params(
            fern_params[transition_idx],
            fern_params[next_transition_idx],
            t_smooth
        )
        
        # Generate fern with interpolated parameters
        x, y = generate_fern_fast(current_params, n_points)
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                           range=[[-3, 3], [0, 11]])
        
        # Apply logarithmic scaling
        H = np.log(H + 1)
        
        # Change colormap for each transition for variety
        cmap_idx = transition_idx % len(cmaps)
        im.set_cmap(cmaps[cmap_idx])
        
        # Update image
        im.set_data(H.T)
        im.set_clim(vmin=0, vmax=np.max(H))
        
        progress = (frame + 1) / frames * 100
        print(f'Frame {frame + 1}/{frames} ({progress:.1f}%) - Colormap: {cmap_idx+1}')
        
        return [im]
    
    # Create animation
    print(f'\n🌿 Creating IMPROVED PSYCHEDELIC Fern animation')
    print(f'Frames: {frames} | FPS: {fps} | Duration: {frames/fps:.1f}s')
    print(f'Smooth transitions: {len(fern_params)-1}')
    print(f'Points per frame: {n_points:,}')
    print(f'Resolution: {bins}x{bins}')
    print(f'Color schemes: {len(cmaps)}\n')
    
    anim = FuncAnimation(fig, update, frames=frames, 
                        interval=1000/fps, blit=True)
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
    # Determine output format
    if filename.endswith('.mp4') and not has_ffmpeg:
        print('⚠️  FFmpeg not found - saving as GIF instead')
        output_path = output_path.replace('.mp4', '.gif')
    
    # Save animation
    print(f'💾 Saving animation...\n')
    if output_path.endswith('.mp4') and has_ffmpeg:
        writer = FFMpegWriter(fps=fps, bitrate=5000, codec='libx264')
        anim.save(output_path, writer=writer, dpi=80)
    else:
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=80)
    
    plt.close()
    
    print(f'\n✨ Animation complete!')
    print(f'📁 Saved to: {output_path}')
    print(f'⏱️  Duration: {frames/fps:.1f} seconds')
    print(f'🔄 Seamless loop: YES')


if __name__ == '__main__':
    # Generate IMPROVED psychedelic fern animation
    create_fern_animation(
        filename='fern_psychedelic_improved.mp4',
        frames=360,        # 360 frames = 12 seconds
        n_points=40000,    # 40k points for speed
        bins=500,          # 500x500 resolution
        fps=30             # Smooth playback
    )
    
    print('\n' + '='*60)
    print('🌿 PSYCHEDELIC BARNSLEY FERN - IMPROVED COLORS')
    print('='*60)
    print('✓ BLACK background for maximum contrast')
    print('✓ 6 different vibrant color schemes')
    print('✓ Smooth continuous morphing')
    print('✓ Perfect seamless loop')
    print('✓ Color scheme changes with each transition')
    print('='*60)