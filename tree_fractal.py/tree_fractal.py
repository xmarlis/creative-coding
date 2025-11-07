import matplotlib.pyplot as plt
import numpy as np
import os
import random

def draw_tree(ax, x, y, angle, depth, branch_length, colors, asymmetry=0.3):
    if depth == 0:
        leaf_color = colors[depth % len(colors)]
        circle = plt.Circle((x, y), 0.015 * branch_length, color=leaf_color, alpha=0.9)
        ax.add_patch(circle)
        return

    branch_color = colors[depth % len(colors)]
    
    # Add randomness based on asymmetry parameter - ONLY if asymmetry > 0
    if asymmetry > 0:
        angle_variation = random.uniform(-asymmetry * 0.4, asymmetry * 0.4)
        length_variation = random.uniform(1 - asymmetry * 0.3, 1 + asymmetry * 0.15)
    else:
        angle_variation = 0
        length_variation = 1
    
    x2 = x + branch_length * length_variation * np.cos(angle + angle_variation)
    y2 = y + branch_length * length_variation * np.sin(angle + angle_variation)
    
    # Draw branch with glow effect
    ax.plot([x, x2], [y, y2], color=branch_color, linewidth=3 + depth * 0.8, alpha=0.4)
    ax.plot([x, x2], [y, y2], color=branch_color, linewidth=1.5 + depth * 0.4, alpha=0.8)

    # Asymmetric branching angles
    base_left = -np.pi/6
    base_right = np.pi/6
    
    if asymmetry > 0:
        left_offset = random.uniform(base_left - asymmetry * 0.3, base_left + asymmetry * 0.3)
        right_offset = random.uniform(base_right - asymmetry * 0.3, base_right + asymmetry * 0.3)
        left_scale = random.uniform(0.7 - asymmetry * 0.1, 0.8 + asymmetry * 0.05)
        right_scale = random.uniform(0.7 - asymmetry * 0.1, 0.8 + asymmetry * 0.05)
        skip_chance = asymmetry * 0.15
    else:
        left_offset = base_left
        right_offset = base_right
        left_scale = 0.75
        right_scale = 0.75
        skip_chance = 0
    
    # Create branches with probability based on asymmetry
    if random.random() > skip_chance:
        draw_tree(ax, x2, y2, angle + left_offset, depth - 1, 
                 branch_length * left_scale, colors, asymmetry)
    
    if random.random() > skip_chance:
        draw_tree(ax, x2, y2, angle + right_offset, depth - 1, 
                 branch_length * right_scale, colors, asymmetry)

def plot_tree_fractal(depth=9, output_path=None, asymmetry=0.3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_facecolor('black')
    ax.axis('off')
    
    # Psychedelic color palette
    colors = [
        "#ff00ff", "#00ffff", "#ffff00", "#ff0099", 
        "#00ff99", "#9900ff", "#ff9900", "#0099ff"
    ]
    
    # Starting parameters - no randomness for symmetric trees
    if asymmetry > 0:
        start_x = 0.5 + random.uniform(-0.01 * asymmetry, 0.01 * asymmetry)
        start_angle = np.pi/2 + random.uniform(-0.05 * asymmetry, 0.05 * asymmetry)
    else:
        start_x = 0.5
        start_angle = np.pi/2
    
    trunk_y = 0.1
    trunk_len = 0.2
    
    draw_tree(ax, start_x, trunk_y, start_angle, depth, trunk_len, colors, asymmetry)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, facecolor='black')
        print(f"✓ Saved: {output_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fractals")
    os.makedirs(outdir, exist_ok=True)

    print("Creating psychedelic tree variations...")
    
    # Create variations with different symmetry levels
    variations = [
        {"name": "perfectly_symmetric", "asymmetry": 0.0, "seed": 100},
        {"name": "symmetric", "asymmetry": 0.0, "seed": 150},
        {"name": "slightly_asymmetric", "asymmetry": 0.2, "seed": 200},
        {"name": "moderately_asymmetric", "asymmetry": 0.4, "seed": 300},
        {"name": "highly_asymmetric", "asymmetry": 0.6, "seed": 400},
        {"name": "very_asymmetric", "asymmetry": 0.8, "seed": 500},
        {"name": "extremely_asymmetric", "asymmetry": 1.0, "seed": 600},
    ]
    
    for var in variations:
        plot_tree_fractal(
            depth=9,
            output_path=os.path.join(outdir, f"psychedelic_tree_{var['name']}.png"),
            asymmetry=var['asymmetry'],
            seed=var['seed']
        )
    
    print(f"\n✨ Done! All variations saved to {outdir}")
    print(f"Created {len(variations)} variations including symmetric versions")
