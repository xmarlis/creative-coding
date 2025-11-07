import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # (kept; safe if unused)
import random
from pathlib import Path

class AsymmetricFernArt:
    def __init__(self, num_points=100000, seed=None):
        self.num_points = num_points
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.x, self.y = self._generate_fern()
    
    def _generate_fern(self):
        """Generate base Barnsley Fern"""
        x = np.zeros(self.num_points)
        y = np.zeros(self.num_points)
        curr_x, curr_y = 0.0, 0.0
        
        for i in range(self.num_points):
            r = random.random()
            if r < 0.01:
                curr_x = 0.0
                curr_y = 0.16 * curr_y
            elif r < 0.86:
                new_x = 0.85 * curr_x + 0.04 * curr_y
                new_y = -0.04 * curr_x + 0.85 * curr_y + 1.6
                curr_x, curr_y = new_x, new_y
            elif r < 0.93:
                new_x = 0.20 * curr_x - 0.26 * curr_y
                new_y = 0.23 * curr_x + 0.22 * curr_y + 1.6
                curr_x, curr_y = new_x, new_y
            else:
                new_x = -0.15 * curr_x + 0.28 * curr_y
                new_y = 0.26 * curr_x + 0.24 * curr_y + 0.44
                curr_x, curr_y = new_x, new_y
            x[i] = curr_x
            y[i] = curr_y
        return x, y
    
    def _setup_plot(self, size=(16, 16)):
        plt.figure(figsize=size, facecolor='black')
        plt.style.use('dark_background')
        plt.axis('equal')
        plt.axis('off')
    
    def _save_plot(self, filename, output_dir):
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}.png',
                    dpi=300, bbox_inches='tight', facecolor='black')
        plt.style.use('default')
        plt.close()
    
    def _setup_plot_white(self, size=(16, 16)):
        plt.figure(figsize=size, facecolor='white')
        plt.style.use('default')
        plt.axis('equal')
        plt.axis('off')
    
    def _save_plot_white(self, filename, output_dir):
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_organic_scatter(self, output_dir):
        self._setup_plot()
        colors = ['#32CD32', '#FF6B35', '#F7931E', '#FFD23F', '#06FFA5']
        configs = [
            (0, 0, 1.0, 0),
            (3.2, -1.8, 0.4, np.pi/7.3),
            (-2.7, 4.1, 0.6, np.pi/3.7),
            (1.9, 2.3, 0.3, np.pi/5.1),
            (-1.1, -2.6, 0.5, np.pi/8.9),
        ]
        for i, (x_off, y_off, scale, angle) in enumerate(configs):
            x_jitter = x_off + np.random.normal(0, 0.3)
            y_jitter = y_off + np.random.normal(0, 0.3)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = self.x * scale * cos_a - self.y * scale * sin_a + x_jitter
            y_rot = self.x * scale * sin_a + self.y * scale * cos_a + y_jitter
            plt.scatter(x_rot, y_rot, s=0.4 * scale, c=colors[i], alpha=0.8)
        self._save_plot('organic_scatter', output_dir)
    
    def create_wind_blown(self, output_dir):
        self._setup_plot()
        colors = ['#00CED1', '#20B2AA', '#48D1CC', '#40E0D0']
        wind_configs = [
            (0, 0, 1.0, 0, 0),
            (2.5, 0.8, 0.7, 0.3, 1.2),
            (4.8, 1.9, 0.5, 0.6, 2.1),
            (-1.5, 3.2, 0.4, -0.4, 0.8),
        ]
        for i, (x_off, y_off, scale, wind_x, wind_y) in enumerate(wind_configs):
            x_wind = self.x + np.sin(self.y * 2) * wind_x
            y_wind = self.y + np.cos(self.x * 1.5) * wind_y
            x_pos = x_wind * scale + x_off
            y_pos = y_wind * scale + y_off
            plt.scatter(x_pos, y_pos, s=0.3 * scale, c=colors[i], alpha=0.7)
        self._save_plot('wind_blown', output_dir)
    
    def create_gravitational_pull(self, output_dir):
        self._setup_plot()
        
        # Even more crazy psychedelic color palettes
        color_palettes = [
            ['#FF00FF', '#00FFFF', '#FFFF00', '#FF0000'],  # Magenta, cyan, yellow, red
            ['#FF1493', '#00FF00', '#FF4500', '#8A2BE2'],  # Deep pink, lime, red-orange, blue-violet
            ['#DC143C', '#00CED1', '#FFD700', '#9932CC'],  # Crimson, dark turquoise, gold, dark orchid
            ['#FF69B4', '#32CD32', '#FF6347', '#4169E1'],  # Hot pink, lime green, tomato, royal blue
        ]
        
        gravity_sources = [
            (0, 0, 1.0, 0, 0),
            (2, 3, 0.6, 1.5, 2.8),
            (-3, 1, 0.5, -2.8, 1.2),
            (1, -2, 0.4, 0.8, -1.5),
        ]
        
        for i, (x_off, y_off, scale, gx, gy) in enumerate(gravity_sources):
            colors = color_palettes[i]
            
            if i == 0:
                x_pos, y_pos = self.x, self.y
            else:
                dist = np.sqrt((self.x - gx)**2 + (self.y - gy)**2)
                strength = 0.3 / (dist + 0.1)
                x_pull = self.x + (gx - self.x) * strength
                y_pull = self.y + (gy - self.y) * strength
                x_pos = x_pull * scale + x_off
                y_pos = y_pull * scale + y_off
            
            # Split points into color segments
            n_points = len(x_pos)
            segment_size = n_points // len(colors)
            
            for j, color in enumerate(colors):
                start_idx = j * segment_size
                end_idx = start_idx + segment_size if j < len(colors) - 1 else n_points
                
                x_segment = x_pos[start_idx:end_idx]
                y_segment = y_pos[start_idx:end_idx]
                
                size = 0.4 if i == 0 else 0.3 * scale
                alpha = 0.8 if i == 0 else 0.7
                
                plt.scatter(x_segment, y_segment, s=size, c=color, alpha=alpha)
        
        self._save_plot('gravitational_pull', output_dir)
    
    def create_liquid_drops(self, output_dir):
        self._setup_plot()
        colors = ['#0080FF', '#00BFFF', '#87CEEB', '#ADD8E6']
        drop_configs = [
            (0, 0, 1.0, 1.0),
            (-1.8, 4.5, 0.6, 0.8),
            (2.3, 2.1, 0.4, 1.3),
            (-0.5, -1.2, 0.5, 0.6),
        ]
        for i, (x_off, y_off, scale, stretch) in enumerate(drop_configs):
            if i == 0:
                plt.scatter(self.x, self.y, s=0.4, c=colors[i], alpha=0.8)
            else:
                x_liquid = self.x * scale
                y_liquid = self.y * scale * stretch
                curve_factor = np.exp(-((self.x)**2 + (self.y - 2)**2) / 4)
                y_liquid += curve_factor * 0.5
                x_pos = x_liquid + x_off
                y_pos = y_liquid + y_off
                plt.scatter(x_pos, y_pos, s=0.3 * scale, c=colors[i], alpha=0.7)
        self._save_plot('liquid_drops', output_dir)
    
    def create_fractal_explosion(self, output_dir):
        self._setup_plot()
        colors = ['#FF0000', '#FF4500', '#FF8C00', '#FFA500', '#FFFF00']
        fragment_configs = [
            (0, 0, 0.8, 0, 0, 0),
            (3.5, 1.2, 0.3, 2.8, 0.9, np.pi/6),
            (-2.1, 3.8, 0.25, -1.7, 3.2, np.pi/4),
            (1.8, -2.5, 0.2, 1.5, -2.1, np.pi/3),
            (-3.2, -1.1, 0.15, -2.9, -0.8, np.pi/5),
        ]
        for i, (x_off, y_off, scale, vel_x, vel_y, rot) in enumerate(fragment_configs):
            if i == 0:
                x_pos = self.x * scale
                y_pos = self.y * scale
            else:
                cos_r, sin_r = np.cos(rot), np.sin(rot)
                x_rot = self.x * cos_r - self.y * sin_r
                y_rot = self.x * sin_r + self.y * cos_r
                x_streak = x_rot + np.random.normal(0, 0.1, len(x_rot))
                y_streak = y_rot + np.random.normal(0, 0.1, len(y_rot))
                x_pos = x_streak * scale + x_off
                y_pos = y_streak * scale + y_off
            plt.scatter(x_pos, y_pos, s=0.2 + scale * 0.3, c=colors[i], alpha=0.8)
        self._save_plot('fractal_explosion', output_dir)
    
    def create_organic_growth(self, output_dir):
        self._setup_plot()
        growth_colors = ['#228B22', '#32CD32', '#90EE90', '#98FB98', '#F0FFF0']
        growth_stages = [
            (0, 0, 0.4, 0),
            (-0.3, 0.8, 0.6, np.pi/12),
            (0.5, 1.9, 0.8, np.pi/8),
            (-0.2, 3.1, 1.0, np.pi/15),
            (0.8, 4.5, 0.7, np.pi/6),
        ]
        for i, (x_off, y_off, scale, tilt) in enumerate(growth_stages):
            x_organic = x_off + np.sin(i * 1.3) * 0.2
            y_organic = y_off + np.cos(i * 0.8) * 0.1
            if tilt != 0:
                cos_t, sin_t = np.cos(tilt), np.sin(tilt)
                x_tilt = self.x * cos_t - self.y * sin_t
                y_tilt = self.x * sin_t + self.y * cos_t
            else:
                x_tilt, y_tilt = self.x, self.y
            x_pos = x_tilt * scale + x_organic
            y_pos = y_tilt * scale + y_organic
            size = 0.2 + scale * 0.3
            alpha = 0.6 + i * 0.08
            plt.scatter(x_pos, y_pos, s=size, c=growth_colors[i], alpha=alpha)
        self._save_plot('organic_growth', output_dir)
    
    def create_chaotic_scatter(self, output_dir):
        self._setup_plot()
        chaos_colors = ['#FF1493', '#00FF7F', '#1E90FF', '#FF4500', '#DA70D6', '#32CD32']
        for i in range(6):
            x_chaos = np.random.uniform(-4, 4)
            y_chaos = np.random.uniform(-1, 6)
            scale_chaos = np.random.uniform(0.2, 0.7)
            angle_chaos = np.random.uniform(0, 2*np.pi)
            skew_x = np.random.uniform(0.5, 1.5)
            skew_y = np.random.uniform(0.5, 1.5)
            cos_c, sin_c = np.cos(angle_chaos), np.sin(angle_chaos)
            x_rot = self.x * skew_x * cos_c - self.y * skew_y * sin_c
            y_rot = self.x * skew_x * sin_c + self.y * skew_y * cos_c
            x_pos = x_rot * scale_chaos + x_chaos
            y_pos = y_rot * scale_chaos + y_chaos
            size = np.random.uniform(0.2, 0.6)
            alpha = np.random.uniform(0.5, 0.9)
            plt.scatter(x_pos, y_pos, s=size, c=chaos_colors[i], alpha=alpha)
        self._save_plot('chaotic_scatter', output_dir)
    
    def create_insane_tornado(self, output_dir):
        self._setup_plot()
        wild_colors = ['#FF0080', '#00FF80', '#8000FF', '#FF8000', '#0080FF', '#FF4080', '#80FF40', '#FF0040']
        chaos_configs = [
            (0, 0, 1.0, 0, 1.0, 1.0, 0),
            (-5.7, 8.2, 0.2, np.pi/2.1, 3.5, 0.3, 1.2),
            (6.8, -3.1, 0.9, np.pi/1.7, 0.4, 2.8, 0.8),
            (-2.3, 4.9, 0.15, np.pi/3.8, 4.2, 0.2, 2.1),
            (4.5, 2.7, 0.7, np.pi/0.9, 0.6, 1.9, 1.5),
            (-4.2, -2.8, 0.3, np.pi/5.7, 2.1, 0.8, 0.6),
            (3.1, 6.3, 1.1, np.pi/2.9, 0.3, 2.2, 1.8),
            (-1.9, 0.7, 0.6, np.pi/4.3, 1.7, 1.4, 0.9),
        ]
        for i, (x_off, y_off, scale, angle, skew_x, skew_y, chaos) in enumerate(chaos_configs):
            x_chaos = x_off + np.random.normal(0, 1.2 * chaos)
            y_chaos = y_off + np.random.normal(0, 1.2 * chaos)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_skewed = self.x * skew_x
            y_skewed = self.y * skew_y
            x_rot = x_skewed * cos_a - y_skewed * sin_a + x_chaos
            y_rot = x_skewed * sin_a + y_skewed * cos_a + y_chaos
            distances = np.sqrt(x_skewed**2 + y_skewed**2)
            swirl_angle = distances * (0.3 + chaos) + i * 0.9
            tornado_strength = 0.2 + chaos * 0.4
            x_swirl = x_rot + np.sin(swirl_angle) * tornado_strength
            y_swirl = y_rot + np.cos(swirl_angle) * tornado_strength
            explosion_x = np.random.normal(0, 0.3 * chaos, len(x_swirl))
            explosion_y = np.random.normal(0, 0.3 * chaos, len(y_swirl))
            x_final = x_swirl * scale + explosion_x
            y_final = y_swirl * scale + explosion_y
            size = np.random.uniform(0.05, 1.2) * scale
            alpha = np.random.uniform(0.4, 0.95)
            plt.scatter(x_final, y_final, s=size, c=wild_colors[i], alpha=alpha)
        
        self._save_plot('insane_tornado', output_dir)
    
    def create_insane_tornado_white(self, output_dir):
        """Create absolutely insane tornado-like scatter with white background"""
        self._setup_plot_white()
        
        wild_colors = ['#FF0080', '#00FF80', '#8000FF', '#FF8000', '#0080FF', '#FF4080', '#80FF40', '#FF0040']
        chaos_configs = [
            (0, 0, 1.0, 0, 1.0, 1.0, 0),
            (-5.7, 8.2, 0.2, np.pi/2.1, 3.5, 0.3, 1.2),
            (6.8, -3.1, 0.9, np.pi/1.7, 0.4, 2.8, 0.8),
            (-2.3, 4.9, 0.15, np.pi/3.8, 4.2, 0.2, 2.1),
            (4.5, 2.7, 0.7, np.pi/0.9, 0.6, 1.9, 1.5),
            (-4.2, -2.8, 0.3, np.pi/5.7, 2.1, 0.8, 0.6),
            (3.1, 6.3, 1.1, np.pi/2.9, 0.3, 2.2, 1.8),
            (-1.9, 0.7, 0.6, np.pi/4.3, 1.7, 1.4, 0.9),
        ]
        for i, (x_off, y_off, scale, angle, skew_x, skew_y, chaos) in enumerate(chaos_configs):
            x_chaos = x_off + np.random.normal(0, 1.2 * chaos)
            y_chaos = y_off + np.random.normal(0, 1.2 * chaos)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_skewed = self.x * skew_x
            y_skewed = self.y * skew_y
            x_rot = x_skewed * cos_a - y_skewed * sin_a + x_chaos
            y_rot = x_skewed * sin_a + y_skewed * cos_a + y_chaos
            distances = np.sqrt(x_skewed**2 + y_skewed**2)
            swirl_angle = distances * (0.3 + chaos) + i * 0.9
            tornado_strength = 0.2 + chaos * 0.4
            x_swirl = x_rot + np.sin(swirl_angle) * tornado_strength
            y_swirl = y_rot + np.cos(swirl_angle) * tornado_strength
            explosion_x = np.random.normal(0, 0.3 * chaos, len(x_swirl))
            explosion_y = np.random.normal(0, 0.3 * chaos, len(y_swirl))
            x_final = x_swirl * scale + explosion_x
            y_final = y_swirl * scale + explosion_y
            size = np.random.uniform(0.05, 1.2) * scale
            alpha = np.random.uniform(0.4, 0.95)
            plt.scatter(x_final, y_final, s=size, c=wild_colors[i], alpha=alpha)
        
        self._save_plot_white('insane_tornado_white', output_dir)

    def create_violet_dreams(self, output_dir):
        self._setup_plot()
        bright_colors = ['#FF0000', '#00FF00', '#0080FF', '#FFFF00', '#FF8000', '#FF00FF', '#00FFFF']
        dream_configs = [
            (0, 0, 1.0, 0, 1.0, 1.0),
            (-2.8, 4.3, 0.6, np.pi/9, 1.2, 0.8),
            (3.2, 2.1, 0.4, np.pi/7, 0.7, 1.3),
            (-1.1, 6.8, 0.3, np.pi/5, 1.4, 0.6),
            (2.7, -1.4, 0.7, np.pi/11, 0.9, 1.1),
            (-3.5, 1.2, 0.5, np.pi/13, 1.1, 0.9),
            (1.3, 5.6, 0.2, np.pi/8, 1.6, 0.5),
        ]
        for i, (x_off, y_off, scale, angle, stretch_x, stretch_y) in enumerate(dream_configs):
            float_x = x_off + np.sin(i * 1.7) * 0.4
            float_y = y_off + np.cos(i * 1.3) * 0.3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_stretched = self.x * stretch_x
            y_stretched = self.y * stretch_y
            x_rot = x_stretched * cos_a - y_stretched * sin_a + float_x
            y_rot = x_stretched * sin_a + y_stretched * cos_a + float_y
            wave_factor = np.sin(self.y * 3 + i) * 0.1
            x_dreamy = x_rot + wave_factor
            y_dreamy = y_rot
            x_final = x_dreamy * scale
            y_final = y_dreamy * scale
            size = 0.2 + scale * 0.4
            alpha = 0.6 + i * 0.05
            plt.scatter(x_final, y_final, s=size, c=bright_colors[i], alpha=alpha)
            if i < 3:
                plt.scatter(x_final, y_final, s=size * 2, c=bright_colors[i], alpha=0.2)
        self._save_plot('violet_dreams', output_dir)
    
    def create_ultra_detailed_microscope(self, output_dir):
        """Create ultra-detailed microscope view with fine structures"""
        self._setup_plot()
        
        # Fine detail colors with transparency gradients
        detail_colors = ['#00FFFF', '#40E0D0', '#20B2AA', '#008B8B', '#006666']
        
        for i, color in enumerate(detail_colors):
            # Focus on different detail levels
            detail_scale = 1 + i * 0.02
            zoom_factor = 1 + i * 0.1
            
            # Add fine structural details
            x_detail = self.x * detail_scale
            y_detail = self.y * detail_scale
            
            # Create microscopic texture
            micro_noise_x = np.random.normal(0, 0.1, len(x_detail))
            micro_noise_y = np.random.normal(0, 0.1, len(y_detail))
            
            x_final = x_detail + micro_noise_x
            y_final = y_detail + micro_noise_y
            
            # Variable point sizes for depth
            point_sizes = 0.1 + np.random.exponential(0.3, len(x_final))
            alpha_values = 0.3 + i * 0.15
            
            plt.scatter(x_final, y_final, s=point_sizes, c=color, alpha=alpha_values)
            
            # Add connection lines for structure
            if i == 0:  # Only for the base layer
                for j in range(0, len(x_final)-1, 100):
                    if j+1 < len(x_final):
                        plt.plot([x_final[j], x_final[j+1]], [y_final[j], y_final[j+1]], 
                                color=color, alpha=0.2, linewidth=0.3)
        
        self._save_plot('ultra_detailed_microscope', output_dir)
    
    def create_fractal_zoom_details(self, output_dir):
        """Create fractal zoom showing self-similar details"""
        self._setup_plot()
        
        # Colors for different zoom levels
        zoom_colors = ['#FF6B6B', '#FFE66D', '#95E1D3', '#A8E6CF', '#C7CEEA']
        
        for i, color in enumerate(zoom_colors):
            # Create nested zoom levels
            zoom_level = 1 + i * 0.5
            center_x, center_y = np.mean(self.x), np.mean(self.y)
            
            # Focus on a specific region and zoom in
            focus_radius = 2.0 / zoom_level
            mask = ((self.x - center_x)**2 + (self.y - center_y)**2) < focus_radius**2
            
            if np.any(mask):
                x_zoom = self.x[mask]
                y_zoom = self.y[mask]
                
                # Scale up the zoomed region
                x_scaled = (x_zoom - center_x) * zoom_level + center_x
                y_scaled = (y_zoom - center_y) * zoom_level + center_y
                
                # Add fine detail enhancement
                detail_factor = 0.1 / zoom_level
                x_enhanced = x_scaled + np.sin(y_scaled * 20) * detail_factor
                y_enhanced = y_scaled + np.cos(x_scaled * 20) * detail_factor
                
                # Variable sizing based on zoom level
                sizes = (1.0 + i * 0.5) * np.ones(len(x_enhanced))
                
                plt.scatter(x_enhanced, y_enhanced, s=sizes, c=color, alpha=0.8)
        
        self._save_plot('fractal_zoom_details', output_dir)
    
    def create_fiber_optic_details(self, output_dir):
        """Create fiber optic effect showing internal structure"""
        self._setup_plot()
        
        # Fiber optic colors - bright cores with glowing edges
        fiber_colors = ['#FFFFFF', '#FFFF99', '#99FFFF', '#FF99FF', '#99FF99']
        
        for i, color in enumerate(fiber_colors):
            # Create fiber-like structures
            fiber_width = 0.5 + i * 0.1
            
            # Add internal fiber structure
            x_fiber = self.x + np.sin(self.y * 0.5 + i) * fiber_width
            y_fiber = self.y + np.cos(self.x * 0.5 + i) * fiber_width
            
            # Core and cladding effect
            core_size = 0.8 - i * 0.1
            cladding_size = core_size * 2
            
            # Draw cladding (outer layer)
            plt.scatter(x_fiber, y_fiber, s=cladding_size, c=color, alpha=0.3)
            
            # Draw core (inner layer)
            plt.scatter(x_fiber, y_fiber, s=core_size, c=color, alpha=0.9)
            
            # Add light transmission lines
            for j in range(0, len(x_fiber)-1, 50):
                if j+1 < len(x_fiber):
                    plt.plot([x_fiber[j], x_fiber[j+1]], [y_fiber[j], y_fiber[j+1]], 
                            color=color, alpha=0.6, linewidth=0.5)
        
        self._save_plot('fiber_optic_details', output_dir)
    
    def create_crystalline_structure_details(self, output_dir):
        """Create detailed crystalline lattice structure"""
        self._setup_plot()
        
        # Crystal structure colors
        crystal_colors = ['#E6E6FA', '#D8BFD8', '#DDA0DD', '#DA70D6', '#BA55D3']
        
        for i, color in enumerate(crystal_colors):
            # Create crystal lattice points
            lattice_spacing = 0.3 + i * 0.1
            
            # Generate lattice structure overlay
            x_crystal = self.x
            y_crystal = self.y
            
            # Add crystalline ordering
            crystal_factor = np.sin(x_crystal / lattice_spacing) * np.cos(y_crystal / lattice_spacing)
            crystal_mask = crystal_factor > 0.5 - i * 0.1
            
            if np.any(crystal_mask):
                x_lattice = x_crystal[crystal_mask]
                y_lattice = y_crystal[crystal_mask]
                
                # Create crystal facets
                facet_size = 2 + i * 0.5
                
                plt.scatter(x_lattice, y_lattice, s=facet_size, c=color, alpha=0.8, marker='D')
                
                # Add crystal connections
                for j in range(0, len(x_lattice)-1, 20):
                    if j+1 < len(x_lattice):
                        distance = np.sqrt((x_lattice[j+1] - x_lattice[j])**2 + 
                                         (y_lattice[j+1] - y_lattice[j])**2)
                        if distance < lattice_spacing * 2:
                            plt.plot([x_lattice[j], x_lattice[j+1]], 
                                   [y_lattice[j], y_lattice[j+1]], 
                                   color=color, alpha=0.4, linewidth=0.8)
        
        self._save_plot('crystalline_structure_details', output_dir)
    
    def create_neural_network_details(self, output_dir):
        """Create neural network-like detailed connections"""
        self._setup_plot()
        
        # Neural network colors
        neural_colors = ['#FF4081', '#7C4DFF', '#448AFF', '#40C4FF', '#18FFFF']
        
        for i, color in enumerate(neural_colors):
            # Create neural nodes
            node_density = 0.02 + i * 0.01
            node_mask = np.random.random(len(self.x)) < node_density
            
            if np.any(node_mask):
                x_nodes = self.x[node_mask]
                y_nodes = self.y[node_mask]
                
                # Node sizes based on "activation"
                activation = np.sin(x_nodes + y_nodes + i) + 1
                node_sizes = 5 + activation * 10
                
                plt.scatter(x_nodes, y_nodes, s=node_sizes, c=color, alpha=0.8)
                
                # Create synaptic connections
                connection_threshold = 1.5 + i * 0.5
                
                for j in range(len(x_nodes)):
                    for k in range(j+1, len(x_nodes)):
                        distance = np.sqrt((x_nodes[k] - x_nodes[j])**2 + 
                                         (y_nodes[k] - y_nodes[j])**2)
                        
                        if distance < connection_threshold:
                            # Connection strength based on distance
                            strength = 1 - (distance / connection_threshold)
                            alpha = strength * 0.6
                            linewidth = strength * 2
                            
                            plt.plot([x_nodes[j], x_nodes[k]], [y_nodes[j], y_nodes[k]], 
                                   color=color, alpha=alpha, linewidth=linewidth)
        
        self._save_plot('neural_network_details', output_dir)
    
    def create_quantum_dot_details(self, output_dir):
        """Create quantum dot array showing fine particle details"""
        self._setup_plot()
        
        # Quantum dot colors with energy levels
        quantum_colors = ['#FF0066', '#FF3366', '#FF6666', '#FF9966', '#FFCC66']
        
        for i, color in enumerate(quantum_colors):
            # Create quantum dot arrays
            dot_spacing = 0.2 + i * 0.05
            energy_level = i + 1
            
            # Quantize positions to create dot array
            x_quantized = np.round(self.x / dot_spacing) * dot_spacing
            y_quantized = np.round(self.y / dot_spacing) * dot_spacing
            
            # Add quantum fluctuations
            fluctuation = 0.05 / energy_level
            x_quantum = x_quantized + np.random.normal(0, fluctuation, len(x_quantized))
            y_quantum = y_quantized + np.random.normal(0, fluctuation, len(y_quantized))
            
            # Dot sizes based on energy states
            dot_sizes = 0.5 + np.sin(x_quantum + y_quantum + energy_level) * 0.3
            
            plt.scatter(x_quantum, y_quantum, s=dot_sizes, c=color, alpha=0.9)
            
            # Add quantum tunneling connections
            tunnel_prob = 0.001 * energy_level
            tunnel_mask = np.random.random(len(x_quantum)) < tunnel_prob
            
            if np.any(tunnel_mask):
                x_tunnel = x_quantum[tunnel_mask]
                y_tunnel = y_quantum[tunnel_mask]
                
                for j in range(len(x_tunnel)):
                    # Random tunneling direction
                    tunnel_angle = np.random.uniform(0, 2*np.pi)
                    tunnel_length = dot_spacing * 2
                    
                    x_end = x_tunnel[j] + tunnel_length * np.cos(tunnel_angle)
                    y_end = y_tunnel[j] + tunnel_length * np.sin(tunnel_angle)
                    
                    plt.plot([x_tunnel[j], x_end], [y_tunnel[j], y_end], 
                           color=color, alpha=0.4, linewidth=1, linestyle='--')
        
        self._save_plot('quantum_dot_details', output_dir)

def main():
    # If the script is already inside a folder named after the script stem,
    # use that folder. Otherwise create one next to the script.
    stem = Path(__file__).stem
    parent = Path(__file__).parent
    if parent.name == stem:
        project_folder = parent
    else:
        project_folder = parent / stem

    images_folder = project_folder / 'images'
    images_folder.mkdir(parents=True, exist_ok=True)

    # No extra tip needed when folder selection is automatic
    print(f"Using image folder: {images_folder}")

    print("Generating Asymmetric Fern Art variations...")
    art = AsymmetricFernArt(num_points=120_000, seed=42)
    variations = [
        ('Organic Scatter', art.create_organic_scatter),
        ('Wind Blown', art.create_wind_blown),
        ('Gravitational Pull', art.create_gravitational_pull),
        ('Liquid Drops', art.create_liquid_drops),
        ('Fractal Explosion', art.create_fractal_explosion),
        ('Organic Growth', art.create_organic_growth),
        ('Chaotic Scatter', art.create_chaotic_scatter),
        ('Insane Tornado', art.create_insane_tornado),
        ('Insane Tornado White', art.create_insane_tornado_white),
        ('Violet Dreams', art.create_violet_dreams),
        ('Ultra Detailed Microscope', art.create_ultra_detailed_microscope),
        ('Fractal Zoom Details', art.create_fractal_zoom_details),
        ('Fiber Optic Details', art.create_fiber_optic_details),
        ('Crystalline Structure Details', art.create_crystalline_structure_details),
        ('Quantum Dot Details', art.create_quantum_dot_details),
    ]
    
    for name, method in variations:
        # Check if file already exists
        filename = method.__name__.replace('create_', '') + '.png'
        filepath = images_folder / filename
        
        if filepath.exists():
            print(f"Skipping {name} - file already exists")
        else:
            print(f"Creating {name}...")
            method(images_folder)
    
    print(f"\nAll Asymmetric Fern Art saved to: {images_folder}")
    print("Generated variations:")
    for name, _ in variations:
        print(f"- {name}")

if __name__ == "__main__":
    main()
