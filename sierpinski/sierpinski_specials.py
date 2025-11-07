import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def sierpinski_colored(points=50000, iterations=150000, colors=None, output_path=None):
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    x, y = 0.5, 0.5
    x_points, y_points, color_idx = [], [], []
    if colors is None:
        colors = ['#e91e63', '#2196f3', '#ffeb3b']  # bright colors that show on black
    for i in range(iterations):
        j = np.random.randint(0, 3)
        x = (x + vertices[j, 0]) / 2
        y = (y + vertices[j, 1]) / 2
        if i >= iterations - points:
            x_points.append(x)
            y_points.append(y)
            color_idx.append(j)
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    for k in range(3):
        mask = np.array(color_idx) == k
        # IMPROVED: much larger points, maximum alpha for maximum visibility
        ax.scatter(np.array(x_points)[mask], np.array(y_points)[mask],
                   s=3.0, c=colors[k], marker='.', alpha=1.0, edgecolors='none', rasterized=True)
    ax.axis('off')
    ax.set_aspect('equal')
    if output_path:
        # IMPROVED: Higher DPI for crisper output
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), dpi=400)
    else:
        plt.show()
    plt.close(fig)

def sierpinski_rotated(points=50000, iterations=150000, angle_deg=30, output_path=None):
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    vertices = vertices @ rot
    x, y = 0.5, 0.5
    x_points, y_points = [], []
    for i in range(iterations):
        j = np.random.randint(0, 3)
        x = (x + vertices[j, 0]) / 2
        y = (y + vertices[j, 1]) / 2
        if i >= iterations - points:
            x_points.append(x)
            y_points.append(y)
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    # IMPROVED: much larger points, maximum alpha
    ax.scatter(x_points, y_points, s=3.0, c='white', marker='.', alpha=1.0, edgecolors='none', rasterized=True)
    ax.axis('off')
    ax.set_aspect('equal')
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), dpi=400)
    else:
        plt.show()
    plt.close(fig)

def sierpinski_perturbed(points=50000, iterations=150000, perturb=0.08, output_path=None):
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    vertices += np.random.uniform(-perturb, perturb, vertices.shape)
    x, y = 0.5, 0.5
    x_points, y_points = [], []
    for i in range(iterations):
        j = np.random.randint(0, 3)
        x = (x + vertices[j, 0]) / 2
        y = (y + vertices[j, 1]) / 2
        if i >= iterations - points:
            x_points.append(x)
            y_points.append(y)
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    # IMPROVED: much larger points, maximum alpha
    ax.scatter(x_points, y_points, s=3.0, c='white', marker='.', alpha=1.0, edgecolors='none', rasterized=True)
    ax.axis('off')
    ax.set_aspect('equal')
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), dpi=400)
    else:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # create a single timestamped output folder next to this script (no nested fixed folder)
    base_dir = os.path.dirname(__file__)
    run_dir = os.path.join(base_dir, datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    print("Generating colored Sierpinski triangle...")
    sierpinski_colored(output_path=os.path.join(run_dir, "sierpinski_colored.png"))
    print("Generating rotated Sierpinski triangle...")
    sierpinski_rotated(angle_deg=45, output_path=os.path.join(run_dir, "sierpinski_rotated.png"))
    print("Generating perturbed Sierpinski triangle...")
    sierpinski_perturbed(output_path=os.path.join(run_dir, "sierpinski_perturbed.png"))
    print(f"Saved images to {run_dir}")