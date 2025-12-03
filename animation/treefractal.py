#!/usr/bin/env python3
"""
Tree Fractal Growth Animation - Psychedelic
Watch the tree grow from a single branch, building layer by layer
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
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


class GrowingTreeFractal:
    """Generate tree fractals with growth animation capability"""
    
    def __init__(self):
        self.all_branches = []
        self.all_colors = []
    
    def draw_branch(self, x, y, angle, length, depth, max_depth, 
                   branch_angle, length_ratio, thickness_ratio, current_depth_limit):
        """
        Recursively draw tree branches up to current_depth_limit
        
        Parameters:
        - current_depth_limit: only draw branches up to this depth (for growth animation)
        """
        if depth > max_depth or depth > current_depth_limit or length < 1:
            return
        
        # Calculate end point of current branch
        x_end = x + length * np.cos(angle)
        y_end = y + length * np.sin(angle)
        
        # Calculate color based on depth (for psychedelic effect)
        color_value = depth / max_depth
        
        # Calculate thickness based on depth
        thickness = max(0.5, 10 * (thickness_ratio ** depth))
        
        # Store branch information with its depth
        self.all_branches.append([(x, y), (x_end, y_end), depth])
        self.all_colors.append((color_value, thickness))
        
        # Recursively draw child branches
        if depth < max_depth and depth < current_depth_limit:
            # Left branch
            self.draw_branch(x_end, y_end, angle + branch_angle, 
                           length * length_ratio, depth + 1, max_depth,
                           branch_angle, length_ratio, thickness_ratio, current_depth_limit)
            
            # Right branch
            self.draw_branch(x_end, y_end, angle - branch_angle, 
                           length * length_ratio, depth + 1, max_depth,
                           branch_angle, length_ratio, thickness_ratio, current_depth_limit)
    
    def generate(self, start_x, start_y, initial_length, initial_angle,
                max_depth, branch_angle, length_ratio, thickness_ratio, 
                current_depth_limit):
        """Generate the tree fractal up to current_depth_limit"""
        self.all_branches = []
        self.all_colors = []
        
        self.draw_branch(start_x, start_y, initial_angle, initial_length, 
                        0, max_depth, branch_angle, length_ratio, thickness_ratio,
                        current_depth_limit)
        
        return self.all_branches, self.all_colors


def create_growing_tree_animation(filename='tree_growing.mp4',
                                 frames=240,
                                 fps=30):
    """
    Create psychedelic tree fractal growth animation
    """
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    print(f'Output will be saved to: {output_path}')
    
    # Tree configuration - single beautiful tree
    max_depth = 12
    branch_angle = np.pi / 6  # 30 degrees
    length_ratio = 0.67
    thickness_ratio = 0.7
    initial_angle = np.pi / 2
    initial_length = 150
    
    # Animation parameters
    # Divide animation into: growth phase + pause + fade out + pause
    growth_frames = int(frames * 0.6)  # 60% for growth
    pause_frames = int(frames * 0.15)  # 15% pause at full tree
    fade_frames = int(frames * 0.2)    # 20% for fade/reset
    final_pause = frames - growth_frames - pause_frames - fade_frames
    
    # Create psychedelic colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        '#8000FF',  # Purple (start from roots)
        '#0000FF',  # Blue
        '#0080FF',  # Sky blue
        '#00FFFF',  # Cyan
        '#00FF80',  # Spring green
        '#00FF00',  # Green
        '#80FF00',  # Lime
        '#FFFF00',  # Yellow
        '#FF8000',  # Orange
        '#FF0000',  # Red
        '#FF0080',  # Hot pink
        '#FF00FF',  # Magenta (tips)
    ]
    cmap = LinearSegmentedColormap.from_list('psychedelic', colors, N=256)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(-400, 400)
    ax.set_ylim(-50, 750)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Initialize tree generator
    tree = GrowingTreeFractal()
    
    def update(frame):
        """Update function for growth animation"""
        ax.clear()
        ax.set_facecolor('black')
        ax.set_xlim(-400, 400)
        ax.set_ylim(-50, 750)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Determine animation phase and progress
        if frame < growth_frames:
            # GROWTH PHASE - tree grows from trunk to full size
            phase = "Growing"
            t = frame / growth_frames
            t_smooth = smootherstep(t)
            
            # Gradually increase depth limit (grows layer by layer)
            current_depth_limit = t_smooth * max_depth
            alpha = 1.0
            
        elif frame < growth_frames + pause_frames:
            # PAUSE PHASE - show full tree
            phase = "Full Tree"
            current_depth_limit = max_depth
            alpha = 1.0
            
        elif frame < growth_frames + pause_frames + fade_frames:
            # FADE PHASE - fade out to restart
            phase = "Fading"
            current_depth_limit = max_depth
            t = (frame - growth_frames - pause_frames) / fade_frames
            alpha = 1.0 - smootherstep(t)
            
        else:
            # FINAL PAUSE - black screen before loop
            phase = "Resetting"
            current_depth_limit = 0
            alpha = 0.0
        
        # Generate tree with current depth limit
        branches, branch_colors = tree.generate(
            start_x=0,
            start_y=0,
            initial_length=initial_length,
            initial_angle=initial_angle,
            max_depth=max_depth,
            branch_angle=branch_angle,
            length_ratio=length_ratio,
            thickness_ratio=thickness_ratio,
            current_depth_limit=current_depth_limit
        )
        
        # Draw all branches with psychedelic colors and current alpha
        for (start, end, depth), (color_val, thickness) in zip(branches, branch_colors):
            # Map color value to psychedelic colormap
            color = cmap(color_val)
            
            # Apply alpha for fade effect
            color_with_alpha = (*color[:3], alpha)
            
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=color_with_alpha, linewidth=thickness, 
                   solid_capstyle='round', antialiased=True)
        
        progress = (frame + 1) / frames * 100
        print(f'Frame {frame + 1}/{frames} ({progress:.1f}%) - '
              f'Phase: {phase:12s} | Depth: {current_depth_limit:.1f}/{max_depth} | '
              f'Branches: {len(branches):4d} | Alpha: {alpha:.2f}')
        
        return ax.get_children()
    
    # Create animation
    print(f'\n🌱 Creating GROWING Tree Fractal animation')
    print(f'Frames: {frames} | FPS: {fps} | Duration: {frames/fps:.1f}s')
    print(f'Max Depth: {max_depth} | Growth: {growth_frames} frames\n')
    
    anim = FuncAnimation(fig, update, frames=frames, 
                        interval=1000/fps, blit=False)
    
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
    print(f'🔄 Loops: Growth → Full tree → Fade → Repeat')


if __name__ == '__main__':
    # Generate growing tree fractal animation
    create_growing_tree_animation(
        filename='tree_growing.mp4',
        frames=240,      # 8 seconds at 30fps
        fps=30           # Smooth playback
    )
    
    print('\n' + '='*60)
    print('🌱 GROWING TREE FRACTAL - PSYCHEDELIC GROWTH')
    print('='*60)
    print('✓ Starts from single trunk')
    print('✓ Grows layer by layer smoothly')
    print('✓ Vibrant rainbow colors (purple roots → magenta tips)')
    print('✓ Full tree display pause')
    print('✓ Smooth fade out and reset')
    print('✓ Perfect loop animation')
    print('='*60)