#!/usr/bin/env python3
"""
Fern Fractal Growth Animation - Psychedelic
Watch a beautiful fern grow using the Barnsley Fern algorithm
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


class BarsleyFern:
    """
    Generate Barnsley Fern using Iterated Function System (IFS)
    
    The fern is created by randomly applying one of four affine transformations:
    1. Stem (1% probability)
    2. Small leaflet - left (7%)
    3. Small leaflet - right (7%)  
    4. Main branch (85%)
    """
    
    def __init__(self):
        self.points = []
    
    def generate(self, n_points, style='classic'):
        """
        Generate fern points using IFS
        
        Parameters:
        - n_points: number of points to generate
        - style: 'classic', 'tailed', 'mutant', 'cyclosorus'
        """
        # Different transformation matrices for different fern styles
        transforms = {
            'classic': [
                # [a, b, c, d, e, f, probability]
                [0.00,  0.00, 0.00,  0.16, 0.00, 0.00, 0.01],  # Stem
                [0.85,  0.04, -0.04, 0.85, 0.00, 1.60, 0.85],  # Main branch
                [0.20, -0.26, 0.23,  0.22, 0.00, 1.60, 0.07],  # Left leaflet
                [-0.15, 0.28, 0.26,  0.24, 0.00, 0.44, 0.07],  # Right leaflet
            ],
            'tailed': [
                [0.00,  0.00, 0.00,  0.25, 0.00, -0.40, 0.02],
                [0.95,  0.005, -0.005, 0.93, -0.002, 0.50, 0.84],
                [0.035, -0.20, 0.16,  0.04, -0.09, 0.02, 0.07],
                [-0.04, 0.20, 0.16,  0.04, 0.083, 0.12, 0.07],
            ],
            'mutant': [
                [0.00,  0.00, 0.00,  0.20, 0.00, -0.12, 0.01],
                [0.845, 0.035, -0.035, 0.82, 0.00, 1.60, 0.85],
                [0.20, -0.31, 0.255,  0.245, 0.00, 0.29, 0.07],
                [-0.15, 0.24, 0.25,  0.20, 0.00, 0.68, 0.07],
            ],
            'cyclosorus': [
                [0.00,  0.00, 0.00,  0.25, 0.00, -0.40, 0.02],
                [0.95,  0.005, -0.005, 0.93, -0.002, 0.50, 0.84],
                [0.035, -0.11, 0.27,  0.01, -0.05, 0.005, 0.07],
                [-0.04, 0.11, 0.27,  0.01, 0.047, 0.060, 0.07],
            ]
        }
        
        params = transforms[style]
        
        # Initialize
        x, y = 0, 0
        points = []
        
        # Generate points
        for _ in range(n_points):
            # Choose transformation based on probabilities
            rand = np.random.random()
            
            if rand < params[0][6]:
                # Transformation 1 (stem)
                a, b, c, d, e, f = params[0][:6]
            elif rand < params[0][6] + params[1][6]:
                # Transformation 2 (main branch)
                a, b, c, d, e, f = params[1][:6]
            elif rand < params[0][6] + params[1][6] + params[2][6]:
                # Transformation 3 (left leaflet)
                a, b, c, d, e, f = params[2][:6]
            else:
                # Transformation 4 (right leaflet)
                a, b, c, d, e, f = params[3][:6]
            
            # Apply transformation
            x_new = a * x + b * y + e
            y_new = c * x + d * y + f
            
            x, y = x_new, y_new
            points.append([x, y])
        
        self.points = np.array(points)
        return self.points


def create_fern_animation(filename='fern_growing.mp4',
                         frames=240,
                         fps=30,
                         n_points=100000):
    """
    Create psychedelic fern fractal growth animation
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    print(f'Output will be saved to: {output_path}')
    
    # Generate full fern first
    fern = BarsleyFern()
    print(f'Generating fern with {n_points:,} points...')
    all_points = fern.generate(n_points, style='classic')
    
    # Get y-coordinate range for growth animation
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    # Animation phases
    growth_frames = int(frames * 0.7)   # 70% for growth
    pause_frames = int(frames * 0.15)   # 15% pause
    fade_frames = int(frames * 0.15)    # 15% fade
    
    # Create psychedelic colormap (bottom to top gradient)
    colors = [
        '#4B0082',  # Indigo (roots)
        '#0000FF',  # Blue
        '#0080FF',  # Sky blue
        '#00FFFF',  # Cyan
        '#00FF80',  # Spring green
        '#00FF00',  # Green
        '#80FF00',  # Lime
        '#FFFF00',  # Yellow
        '#FFD700',  # Gold
        '#FFA500',  # Orange
        '#FF6347',  # Tomato
        '#FF1493',  # Deep pink (tips)
    ]
    cmap = LinearSegmentedColormap.from_list('fern', colors, N=256)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 12), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=0.5, c=[], cmap=cmap, alpha=1.0)
    
    def update(frame):
        """Update function for growth animation"""
        # Determine animation phase
        if frame < growth_frames:
            # GROWTH PHASE
            phase = "Growing"
            t = frame / growth_frames
            t_smooth = smootherstep(t)
            
            # Calculate current height threshold (grows from bottom to top)
            current_y_threshold = y_min + t_smooth * (y_max - y_min)
            
            # Filter points up to current height
            mask = all_points[:, 1] <= current_y_threshold
            visible_points = all_points[mask]
            
            alpha = 1.0
            
        elif frame < growth_frames + pause_frames:
            # PAUSE PHASE
            phase = "Full"
            visible_points = all_points
            alpha = 1.0
            
        else:
            # FADE PHASE
            phase = "Fading"
            visible_points = all_points
            t = (frame - growth_frames - pause_frames) / fade_frames
            alpha = 1.0 - smootherstep(t)
        
        if len(visible_points) > 0:
            # Normalize y-coordinates for color mapping (0 to 1)
            colors_normalized = (visible_points[:, 1] - y_min) / (y_max - y_min)
            
            # Update scatter plot
            scatter.set_offsets(visible_points)
            scatter.set_array(colors_normalized)
            scatter.set_alpha(alpha)
            
            # Auto-adjust view to fit fern
            if frame == 0:
                x_min, x_max = visible_points[:, 0].min(), visible_points[:, 0].max()
                x_margin = (x_max - x_min) * 0.1
                y_margin = (y_max - y_min) * 0.05
                
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        progress = (frame + 1) / frames * 100
        print(f'Frame {frame + 1}/{frames} ({progress:.1f}%) - '
              f'Phase: {phase:8s} | Points: {len(visible_points):6d}/{len(all_points):6d} | '
              f'Alpha: {alpha:.2f}')
        
        return [scatter]
    
    # Create animation
    print(f'\n🌿 Creating PSYCHEDELIC Fern Fractal animation')
    print(f'Frames: {frames} | FPS: {fps} | Duration: {frames/fps:.1f}s')
    print(f'Total points: {n_points:,}\n')
    
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
        anim.save(output_path, writer=writer, dpi=100)
    else:
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
    
    plt.close()
    
    print(f'\n✨ Animation complete!')
    print(f'📁 Saved to: {output_path}')
    print(f'⏱️  Duration: {frames/fps:.1f} seconds')
    print(f'🔄 Loop: Growth → Full fern → Fade → Repeat')


if __name__ == '__main__':
    # Generate growing fern fractal animation
    create_fern_animation(
        filename='fern_growing.mp4',
        frames=240,         # 8 seconds at 30fps
        fps=30,             # Smooth playback
        n_points=100000     # Number of points (more = more detail)
    )
    
    print('\n' + '='*60)
    print('🌿 BARNSLEY FERN FRACTAL - PSYCHEDELIC GROWTH')
    print('='*60)
    print('✓ Grows from bottom (roots) to top (fronds)')
    print('✓ 100,000 points for detailed appearance')
    print('✓ Rainbow gradient: Indigo roots → Pink tips')
    print('✓ Smooth growth animation')
    print('✓ Pause at full fern')
    print('✓ Fade out and perfect loop')
    print('='*60)
    print('\nBarnsley Fern uses an Iterated Function System (IFS)')
    print('with probabilistic transformations to create natural fern shapes!')