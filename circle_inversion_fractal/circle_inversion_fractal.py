import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from datetime import datetime

class CircleInversionFractal:
    def __init__(self, width=2000, height=2000, max_depth=6):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.circles = []
        
    def invert_circle(self, cx, cy, radius, inv_cx, inv_cy, inv_radius):
        """
        Invert a circle with center (cx, cy) and radius 'radius'
        with respect to inversion circle centered at (inv_cx, inv_cy) with radius 'inv_radius'
        """
        # Distance from inversion center to circle center
        d = np.sqrt((cx - inv_cx)**2 + (cy - inv_cy)**2)
        
        # Avoid division by zero
        if d < 1e-10:
            return None
        
        # Calculate inverted circle parameters
        # If the circle passes through the inversion center, it becomes a line (skip for simplicity)
        if abs(d - radius) < 1e-10:
            return None
            
        # Inversion formula for circles
        k = inv_radius ** 2
        
        # Distance to closest and farthest points on original circle
        d1 = d - radius
        d2 = d + radius
        
        if abs(d1) < 1e-10 or abs(d2) < 1e-10:
            return None
            
        # Inverted distances
        d1_inv = k / d1
        d2_inv = k / d2
        
        # New radius and center
        new_radius = abs(d1_inv - d2_inv) / 2
        d_new = (d1_inv + d2_inv) / 2
        
        # Direction from inversion center to original circle center
        dx = (cx - inv_cx) / d
        dy = (cy - inv_cy) / d
        
        # New center position
        new_cx = inv_cx + dx * d_new
        new_cy = inv_cy + dy * d_new
        
        return (new_cx, new_cy, new_radius)
    
    def generate_apollonian_gasket(self, c1, c2, c3, depth=0):
        """
        Generate Apollonian gasket - a more structured circle inversion fractal
        c1, c2, c3 are tuples of (x, y, radius, curvature)
        """
        if depth > self.max_depth:
            return
            
        # Extract parameters
        x1, y1, r1, k1 = c1
        x2, y2, r2, k2 = c2
        x3, y3, r3, k3 = c3
        
        # Descartes Circle Theorem for curvature
        k4_plus = k1 + k2 + k3 + 2 * np.sqrt(k1*k2 + k2*k3 + k3*k1)
        k4_minus = k1 + k2 + k3 - 2 * np.sqrt(k1*k2 + k2*k3 + k3*k1)
        
        for k4 in [k4_plus, k4_minus]:
            if abs(k4) < 1e-10:
                continue
                
            r4 = 1.0 / abs(k4)
            
            # Complex Descartes Theorem for center
            z1 = complex(x1, y1) * k1
            z2 = complex(x2, y2) * k2
            z3 = complex(x3, y3) * k3
            
            term = np.sqrt(k1*k2*complex(x1-x2, y1-y2)**2 + 
                          k2*k3*complex(x2-x3, y2-y3)**2 + 
                          k3*k1*complex(x3-x1, y3-y1)**2)
            
            z4_plus = (z1 + z2 + z3 + 2*term) / k4
            z4_minus = (z1 + z2 + z3 - 2*term) / k4
            
            for z4 in [z4_plus, z4_minus]:
                x4, y4 = z4.real, z4.imag
                
                # Check if this circle is valid and doesn't overlap too much
                valid = True
                for existing_circle in [(x1,y1,r1), (x2,y2,r2), (x3,y3,r3)]:
                    ex, ey, er = existing_circle
                    dist = np.sqrt((x4-ex)**2 + (y4-ey)**2)
                    if dist < abs(er - r4) * 0.9:  # Too much overlap
                        valid = False
                        break
                
                if valid and r4 > 0.001:  # Minimum radius threshold
                    self.circles.append((x4, y4, r4, depth))
                    
                    # Recurse with new circle
                    if depth < self.max_depth - 1:
                        c4 = (x4, y4, r4, k4)
                        self.generate_apollonian_gasket(c1, c2, c4, depth + 1)
                        self.generate_apollonian_gasket(c1, c3, c4, depth + 1)
                        self.generate_apollonian_gasket(c2, c3, c4, depth + 1)
    
    def generate_simple_inversion(self):
        """
        Generate a simpler circle inversion fractal pattern
        """
        # Start with a large outer circle
        outer_radius = 1.0
        self.circles.append((0, 0, outer_radius, 0))
        
        # Add three mutually tangent circles inside
        r_inner = outer_radius / 3.0
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        
        for i, angle in enumerate(angles):
            x = (outer_radius - r_inner) * np.cos(angle) * 0.5
            y = (outer_radius - r_inner) * np.sin(angle) * 0.5
            self.circles.append((x, y, r_inner, 1))
            
            # Add smaller circles recursively
            self.add_recursive_circles(x, y, r_inner, 2, angle)
    
    def add_recursive_circles(self, cx, cy, radius, depth, base_angle):
        """
        Add circles recursively in a pattern
        """
        if depth > self.max_depth:
            return
            
        num_circles = 6
        scale = 0.35
        
        for i in range(num_circles):
            angle = base_angle + (2 * np.pi * i / num_circles)
            new_radius = radius * scale
            distance = radius - new_radius
            
            new_x = cx + distance * np.cos(angle)
            new_y = cy + distance * np.sin(angle)
            
            self.circles.append((new_x, new_y, new_radius, depth))
            
            if new_radius > 0.005:
                self.add_recursive_circles(new_x, new_y, new_radius, depth + 1, angle)
    
    def plot_fractal(self, color_mode='depth'):
        """
        Plot the fractal with beautiful colors
        """
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Color maps
        if color_mode == 'depth':
            colors = plt.cm.twilight(np.linspace(0, 1, self.max_depth + 1))
        elif color_mode == 'rainbow':
            colors = plt.cm.rainbow(np.linspace(0, 1, self.max_depth + 1))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, self.max_depth + 1))
        
        # Sort circles by size (largest first) for better layering
        sorted_circles = sorted(self.circles, key=lambda c: c[2], reverse=True)
        
        for cx, cy, radius, depth in sorted_circles:
            depth_idx = min(int(depth), self.max_depth)
            circle = Circle((cx, cy), radius, 
                          color=colors[depth_idx],
                          fill=True,
                          alpha=0.7,
                          edgecolor='white',
                          linewidth=0.5)
            ax.add_patch(circle)
        
        plt.tight_layout()
        return fig
    
    def save_fractal(self, filename, output_dir='circle_inversion_output'):
        """
        Save the fractal to a file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        fig = self.plot_fractal()
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"Saved: {filepath}")
        return filepath


    def generate_spiral_pattern(self):
        """
        Generate spiral-based circle pattern
        """
        outer_radius = 1.0
        self.circles.append((0, 0, outer_radius, 0))
        
        num_spirals = 3
        for spiral_idx in range(num_spirals):
            base_angle = (2 * np.pi * spiral_idx) / num_spirals
            self.add_spiral_branch(0, 0, outer_radius * 0.3, base_angle, 1)
    
    def add_spiral_branch(self, cx, cy, radius, angle, depth):
        """
        Add circles in a spiral pattern
        """
        if depth > self.max_depth or radius < 0.005:
            return
        
        # Golden ratio for aesthetics
        golden = 1.618
        scale = 0.6
        angle_increment = np.pi / 3
        
        distance = radius * 1.8
        new_x = cx + distance * np.cos(angle)
        new_y = cy + distance * np.sin(angle)
        new_radius = radius * scale
        
        self.circles.append((new_x, new_y, new_radius, depth))
        
        # Continue spiral
        self.add_spiral_branch(new_x, new_y, new_radius, angle + angle_increment, depth + 1)
        
        # Add side branches
        if depth < self.max_depth - 1:
            for side_angle in [angle + np.pi/2, angle - np.pi/2]:
                side_distance = radius * 1.2
                side_x = cx + side_distance * np.cos(side_angle)
                side_y = cy + side_distance * np.sin(side_angle)
                side_radius = radius * 0.4
                if side_radius > 0.01:
                    self.circles.append((side_x, side_y, side_radius, depth + 1))
    
    def generate_mandala_pattern(self):
        """
        Generate mandala-like circular pattern
        """
        outer_radius = 1.0
        self.circles.append((0, 0, outer_radius, 0))
        
        # Multiple rings
        num_rings = 4
        for ring in range(1, num_rings + 1):
            ring_radius = outer_radius * (0.8 - ring * 0.15)
            num_circles = 6 + ring * 2
            
            for i in range(num_circles):
                angle = (2 * np.pi * i) / num_circles
                circle_radius = outer_radius * 0.12 / ring
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                self.circles.append((x, y, circle_radius, ring))
                
                # Add smaller circles around each main circle
                if ring < num_rings:
                    self.add_mandala_detail(x, y, circle_radius, angle, ring + 1)
    
    def add_mandala_detail(self, cx, cy, radius, base_angle, depth):
        """
        Add detail circles around mandala circles
        """
        if depth > self.max_depth:
            return
        
        num_detail = 5
        detail_radius = radius * 0.3
        
        for i in range(num_detail):
            angle = base_angle + (2 * np.pi * i) / num_detail
            distance = radius * 0.7
            x = cx + distance * np.cos(angle)
            y = cy + distance * np.sin(angle)
            self.circles.append((x, y, detail_radius, depth))
    
    def generate_binary_tree_circles(self):
        """
        Generate circles in a binary tree pattern
        """
        self.circles.append((0, 0, 1.0, 0))
        self.add_binary_branch(0, 0.3, 0.25, -np.pi/2, 1, "left")
        self.add_binary_branch(0, 0.3, 0.25, -np.pi/2, 1, "right")
    
    def add_binary_branch(self, cx, cy, radius, angle, depth, branch_type):
        """
        Add circles in binary tree pattern
        """
        if depth > self.max_depth or radius < 0.01:
            return
        
        self.circles.append((cx, cy, radius, depth))
        
        # Branch parameters
        branch_angle = np.pi / 5
        scale = 0.7
        distance = radius * 2.5
        
        # Left branch
        left_angle = angle - branch_angle
        left_x = cx + distance * np.cos(left_angle)
        left_y = cy + distance * np.sin(left_angle)
        left_radius = radius * scale
        
        # Right branch
        right_angle = angle + branch_angle
        right_x = cx + distance * np.cos(right_angle)
        right_y = cy + distance * np.sin(right_angle)
        right_radius = radius * scale
        
        self.add_binary_branch(left_x, left_y, left_radius, left_angle, depth + 1, "left")
        self.add_binary_branch(right_x, right_y, right_radius, right_angle, depth + 1, "right")
    
    def generate_hexagonal_packing(self):
        """
        Generate hexagonal circle packing
        """
        self.circles.append((0, 0, 1.0, 0))
        
        # Initial hexagon of circles
        hex_radius = 0.28
        for i in range(6):
            angle = (2 * np.pi * i) / 6
            x = 0.6 * np.cos(angle)
            y = 0.6 * np.sin(angle)
            self.circles.append((x, y, hex_radius, 1))
            self.add_hex_cluster(x, y, hex_radius, angle, 2)
    
    def add_hex_cluster(self, cx, cy, radius, base_angle, depth):
        """
        Add hexagonal clusters recursively
        """
        if depth > self.max_depth or radius < 0.01:
            return
        
        new_radius = radius * 0.4
        distance = radius * 0.85
        
        for i in range(6):
            angle = base_angle + (2 * np.pi * i) / 6
            x = cx + distance * np.cos(angle)
            y = cy + distance * np.sin(angle)
            self.circles.append((x, y, new_radius, depth))
            
            if depth < self.max_depth - 1 and new_radius > 0.015:
                self.add_hex_cluster(x, y, new_radius, angle, depth + 1)
    
    def generate_fibonacci_spiral(self):
        """
        Generate circles following Fibonacci spiral
        """
        self.circles.append((0, 0, 1.0, 0))
        
        golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
        
        num_circles = 150
        for i in range(1, num_circles):
            angle = i * golden_angle
            distance = 0.1 * np.sqrt(i)
            radius = 0.05 / np.sqrt(i + 1)
            
            if distance < 0.95 and radius > 0.005:
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                depth = min(int(i / 25), self.max_depth)
                self.circles.append((x, y, radius, depth))
    
    def generate_concentric_waves(self):
        """
        Generate concentric wave patterns
        """
        num_waves = 8
        for wave in range(num_waves):
            wave_radius = 0.95 - (wave * 0.11)
            num_circles = 16 + wave * 4
            circle_size = 0.05 / (wave + 1)
            
            for i in range(num_circles):
                angle = (2 * np.pi * i) / num_circles + (wave * 0.2)
                x = wave_radius * np.cos(angle)
                y = wave_radius * np.sin(angle)
                self.circles.append((x, y, circle_size, wave))
                
                # Add smaller circles
                if circle_size > 0.01:
                    for j in range(3):
                        sub_angle = angle + (2 * np.pi * j) / 3
                        sub_distance = circle_size * 0.8
                        sub_x = x + sub_distance * np.cos(sub_angle)
                        sub_y = y + sub_distance * np.sin(sub_angle)
                        sub_radius = circle_size * 0.3
                        self.circles.append((sub_x, sub_y, sub_radius, wave + 1))
    
    def plot_fractal(self, color_mode='depth'):
        """
        Plot the fractal with beautiful colors
        """
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Color maps
        cmaps = {
            'depth': plt.cm.twilight,
            'rainbow': plt.cm.rainbow,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'cool': plt.cm.cool,
            'spring': plt.cm.spring,
            'ocean': plt.cm.ocean
        }
        
        cmap = cmaps.get(color_mode, plt.cm.twilight)
        colors = cmap(np.linspace(0, 1, self.max_depth + 2))
        
        # Sort circles by size (largest first) for better layering
        sorted_circles = sorted(self.circles, key=lambda c: c[2], reverse=True)
        
        for cx, cy, radius, depth in sorted_circles:
            depth_idx = min(int(depth), self.max_depth)
            circle = Circle((cx, cy), radius, 
                          color=colors[depth_idx],
                          fill=True,
                          alpha=0.7,
                          edgecolor='white',
                          linewidth=0.5)
            ax.add_patch(circle)
        
        plt.tight_layout()
        return fig


def main():
    """
    Generate multiple variations of circle inversion fractals
    """
    print("🎨 Circle Inversion Fractal Generator")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'circle_inversion_output_{timestamp}'
    
    # Configuration sets for different visual styles
    configs = [
        # Organic recursive patterns
        {'max_depth': 5, 'style': 'simple', 'color': 'depth', 
         'name': '01_organic_recursive_basic'},
        {'max_depth': 6, 'style': 'simple', 'color': 'rainbow', 
         'name': '02_organic_recursive_rainbow'},
        {'max_depth': 7, 'style': 'simple', 'color': 'plasma', 
         'name': '03_organic_recursive_plasma'},
        
        # Spiral patterns
        {'max_depth': 8, 'style': 'spiral', 'color': 'cool', 
         'name': '04_spiral_cool'},
        {'max_depth': 9, 'style': 'spiral', 'color': 'ocean', 
         'name': '05_spiral_ocean'},
        
        # Mandala patterns
        {'max_depth': 5, 'style': 'mandala', 'color': 'spring', 
         'name': '06_mandala_spring'},
        {'max_depth': 6, 'style': 'mandala', 'color': 'twilight', 
         'name': '07_mandala_twilight'},
        
        # Binary tree
        {'max_depth': 7, 'style': 'binary', 'color': 'viridis', 
         'name': '08_binary_tree_viridis'},
        {'max_depth': 8, 'style': 'binary', 'color': 'plasma', 
         'name': '09_binary_tree_plasma'},
        
        # Hexagonal packing
        {'max_depth': 5, 'style': 'hexagonal', 'color': 'rainbow', 
         'name': '10_hexagonal_rainbow'},
        {'max_depth': 6, 'style': 'hexagonal', 'color': 'cool', 
         'name': '11_hexagonal_cool'},
        
        # Fibonacci spiral
        {'max_depth': 6, 'style': 'fibonacci', 'color': 'twilight', 
         'name': '12_fibonacci_twilight'},
        {'max_depth': 7, 'style': 'fibonacci', 'color': 'spring', 
         'name': '13_fibonacci_spring'},
        
        # Concentric waves
        {'max_depth': 8, 'style': 'waves', 'color': 'ocean', 
         'name': '14_concentric_waves_ocean'},
        {'max_depth': 9, 'style': 'waves', 'color': 'plasma', 
         'name': '15_concentric_waves_plasma'},
    ]
    
    saved_files = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i:02d}/{len(configs)}] Generating {config['name']}...")
        
        fractal = CircleInversionFractal(max_depth=config['max_depth'])
        
        # Generate different patterns based on style
        if config['style'] == 'simple':
            fractal.generate_simple_inversion()
        elif config['style'] == 'spiral':
            fractal.generate_spiral_pattern()
        elif config['style'] == 'mandala':
            fractal.generate_mandala_pattern()
        elif config['style'] == 'binary':
            fractal.generate_binary_tree_circles()
        elif config['style'] == 'hexagonal':
            fractal.generate_hexagonal_packing()
        elif config['style'] == 'fibonacci':
            fractal.generate_fibonacci_spiral()
        elif config['style'] == 'waves':
            fractal.generate_concentric_waves()
        
        filename = f"{config['name']}.png"
        
        # Save with specified color scheme
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        cmaps = {
            'depth': plt.cm.twilight,
            'rainbow': plt.cm.rainbow,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'cool': plt.cm.cool,
            'spring': plt.cm.spring,
            'ocean': plt.cm.ocean,
            'twilight': plt.cm.twilight
        }
        
        cmap = cmaps.get(config['color'], plt.cm.twilight)
        colors = cmap(np.linspace(0, 1, config['max_depth'] + 2))
        
        sorted_circles = sorted(fractal.circles, key=lambda c: c[2], reverse=True)
        
        for cx, cy, radius, depth in sorted_circles:
            depth_idx = min(int(depth), config['max_depth'])
            circle = Circle((cx, cy), radius, 
                          color=colors[depth_idx],
                          fill=True,
                          alpha=0.7,
                          edgecolor='white',
                          linewidth=0.5)
            ax.add_patch(circle)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        saved_files.append(filepath)
        print(f"  ✓ Created {len(fractal.circles):,} circles | {config['style']} style | {config['color']} colors")
    
    print("\n" + "=" * 60)
    print(f"✨ Generation complete!")
    print(f"📁 All files saved in: {output_dir}/")
    print(f"🖼️  Total images: {len(saved_files)}")
    print("\nPattern types generated:")
    print("  • Organic Recursive (3 variations)")
    print("  • Spiral Patterns (2 variations)")
    print("  • Mandala Patterns (2 variations)")
    print("  • Binary Tree (2 variations)")
    print("  • Hexagonal Packing (2 variations)")
    print("  • Fibonacci Spiral (2 variations)")
    print("  • Concentric Waves (2 variations)")
    
    return output_dir


if __name__ == "__main__":
    output_folder = main()
    print(f"\n🎨 Circle Inversion Fractals ready in '{output_folder}'!")