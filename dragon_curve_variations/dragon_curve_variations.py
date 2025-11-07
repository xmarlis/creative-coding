import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import math
from datetime import datetime

class DragonCurveArt:
    def __init__(self, iterations=12, seed=None):
        self.iterations = iterations
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate base dragon curve
        self.x, self.y = self._generate_dragon_curve()
    
    def _generate_dragon_curve(self):
        """Generate Dragon Curve using L-system"""
        # Start with initial string
        dragon_string = "FX"
        
        # Apply L-system rules for specified iterations
        for _ in range(self.iterations):
            new_string = ""
            for char in dragon_string:
                if char == 'X':
                    new_string += "X+YF+"
                elif char == 'Y':
                    new_string += "-FX-Y"
                else:
                    new_string += char
            dragon_string = new_string
        
        # Convert string to coordinates
        x, y = [0], [0]
        angle = 0
        current_x, current_y = 0, 0
        
        for char in dragon_string:
            if char == 'F':
                # Move forward
                current_x += math.cos(math.radians(angle))
                current_y += math.sin(math.radians(angle))
                x.append(current_x)
                y.append(current_y)
            elif char == '+':
                # Turn right 90 degrees
                angle -= 90
            elif char == '-':
                # Turn left 90 degrees
                angle += 90
        
        return np.array(x), np.array(y)
    
    def _setup_plot(self, size=(16, 16), bg_color='black'):
        plt.figure(figsize=size, facecolor=bg_color)
        if bg_color == 'black':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        plt.axis('equal')
        plt.axis('off')
    
    def _save_plot(self, filename, output_dir, bg_color='black'):
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}.png', 
                   dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.style.use('default')
        plt.close()
    
    def create_rainbow_gradient(self, output_dir):
        """Create rainbow gradient dragon curve"""
        self._setup_plot()
        
        # Create rainbow colors
        n_points = len(self.x)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_points))
        
        plt.scatter(self.x, self.y, c=colors, s=2, alpha=0.8)
        plt.plot(self.x, self.y, color='white', alpha=0.3, linewidth=0.5)
        
        self._save_plot('dragon_rainbow_gradient', output_dir)
    
    def create_neon_segments(self, output_dir):
        """Create neon-colored segmented dragon curve"""
        self._setup_plot()
        
        # Bright neon colors
        neon_colors = ['#FF0080', '#00FF80', '#8000FF', '#FF8000', '#00FFFF', '#FFFF00', '#FF4080']
        
        # Divide curve into segments
        n_points = len(self.x)
        segment_size = n_points // len(neon_colors)
        
        for i, color in enumerate(neon_colors):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < len(neon_colors) - 1 else n_points
            
            x_segment = self.x[start_idx:end_idx]
            y_segment = self.y[start_idx:end_idx]
            
            plt.plot(x_segment, y_segment, color=color, linewidth=3, alpha=0.9)
            plt.scatter(x_segment[::10], y_segment[::10], c=color, s=10, alpha=0.7)
        
        self._save_plot('dragon_neon_segments', output_dir)
    
    def create_spiral_view(self, output_dir):
        """Create spiral transformation of dragon curve"""
        self._setup_plot()
        
        # Apply spiral transformation
        center_x, center_y = np.mean(self.x), np.mean(self.y)
        spiral_colors = ['#FF1493', '#00CED1', '#32CD32', '#FFD700', '#FF4500']
        
        for i, color in enumerate(spiral_colors):
            spiral_factor = (i + 1) * 0.3
            angle_offset = i * np.pi / 3
            
            # Transform coordinates
            dist = np.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)
            angles = np.arctan2(self.y - center_y, self.x - center_x)
            
            new_angles = angles + dist * spiral_factor + angle_offset
            new_dist = dist * (1 + i * 0.1)
            
            x_spiral = center_x + new_dist * np.cos(new_angles)
            y_spiral = center_y + new_dist * np.sin(new_angles)
            
            plt.plot(x_spiral, y_spiral, color=color, alpha=0.7, linewidth=2)
        
        self._save_plot('dragon_spiral_view', output_dir)
    
    def create_fractal_explosion(self, output_dir):
        """Create exploding fractal effect"""
        self._setup_plot()
        
        explosion_colors = ['#FF0000', '#FF4500', '#FF8C00', '#FFD700', '#ADFF2F', '#00FF7F', '#00CED1']
        
        for i, color in enumerate(explosion_colors):
            # Apply explosion transformation
            explosion_strength = (i + 1) * 0.15
            angle_spread = i * np.pi / 4
            
            # Calculate distance from center
            center_x, center_y = np.mean(self.x), np.mean(self.y)
            dist = np.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)
            
            # Explosive displacement
            angles = np.arctan2(self.y - center_y, self.x - center_x) + angle_spread
            explosion_factor = 1 + explosion_strength * (dist / np.max(dist))
            
            x_explode = center_x + (self.x - center_x) * explosion_factor
            y_explode = center_y + (self.y - center_y) * explosion_factor
            
            # Add random scatter
            x_explode += np.random.normal(0, explosion_strength, len(x_explode))
            y_explode += np.random.normal(0, explosion_strength, len(y_explode))
            
            plt.scatter(x_explode, y_explode, c=color, s=1.5, alpha=0.8)
        
        self._save_plot('dragon_fractal_explosion', output_dir)
    
    def create_wave_distortion(self, output_dir):
        """Create wave-distorted dragon curve"""
        self._setup_plot()
        
        wave_colors = ['#FF69B4', '#00BFFF', '#98FB98', '#DDA0DD', '#F0E68C']
        
        for i, color in enumerate(wave_colors):
            wave_freq = (i + 1) * 0.02
            wave_amp = (i + 1) * 5
            phase_offset = i * np.pi / 2
            
            # Apply wave distortion
            x_wave = self.x + wave_amp * np.sin(self.y * wave_freq + phase_offset)
            y_wave = self.y + wave_amp * np.cos(self.x * wave_freq + phase_offset)
            
            plt.plot(x_wave, y_wave, color=color, alpha=0.7, linewidth=2)
            
            # Add highlight points
            highlight_indices = np.arange(0, len(x_wave), 50)
            plt.scatter(x_wave[highlight_indices], y_wave[highlight_indices], 
                       c=color, s=20, alpha=0.9)
        
        self._save_plot('dragon_wave_distortion', output_dir)
    
    def create_psychedelic_layers(self, output_dir):
        """Create psychedelic multi-layered effect"""
        self._setup_plot()
        
        psychedelic_colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF0080', '#80FF00', '#0080FF']
        
        for i, color in enumerate(psychedelic_colors):
            # Apply different transformations per layer
            scale = 1 + i * 0.1
            rotation = i * np.pi / 6
            offset_x = i * 10 * np.sin(i)
            offset_y = i * 10 * np.cos(i)
            
            # Transform coordinates
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            x_rot = self.x * cos_r - self.y * sin_r
            y_rot = self.x * sin_r + self.y * cos_r
            
            x_final = x_rot * scale + offset_x
            y_final = y_rot * scale + offset_y
            
            # Add psychedelic wave effect
            x_final += 5 * np.sin(y_final * 0.05 + i)
            y_final += 5 * np.cos(x_final * 0.05 + i)
            
            plt.plot(x_final, y_final, color=color, alpha=0.6, linewidth=1.5)
            
            # Add glowing effect
            plt.plot(x_final, y_final, color=color, alpha=0.3, linewidth=4)
        
        self._save_plot('dragon_psychedelic_layers', output_dir)
    
    def create_psychedelic_layers_white_bg(self, output_dir):
        """Create psychedelic multi-layered effect on white background"""
        self._setup_plot(bg_color='white')
        
        # Bright, saturated colors that work well on white
        psychedelic_colors = ['#FF0080', '#8000FF', '#0080FF', '#00FF80', '#FF8000', '#FF0040']
        
        for i, color in enumerate(psychedelic_colors):
            # Apply different transformations per layer
            scale = 1 + i * 0.1
            rotation = i * np.pi / 6
            offset_x = i * 10 * np.sin(i)
            offset_y = i * 10 * np.cos(i)
            
            # Transform coordinates
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            x_rot = self.x * cos_r - self.y * sin_r
            y_rot = self.x * sin_r + self.y * cos_r
            
            x_final = x_rot * scale + offset_x
            y_final = y_rot * scale + offset_y
            
            # Add psychedelic wave effect
            x_final += 5 * np.sin(y_final * 0.05 + i)
            y_final += 5 * np.cos(x_final * 0.05 + i)
            
            # Thicker lines for better visibility on white
            plt.plot(x_final, y_final, color=color, alpha=0.8, linewidth=2.5)
            
            # Add accent points instead of glow effect
            accent_indices = np.arange(0, len(x_final), 30)
            plt.scatter(x_final[accent_indices], y_final[accent_indices], 
                       c=color, s=8, alpha=0.9)
        
        self._save_plot('dragon_psychedelic_layers_white_bg', output_dir, bg_color='white')
    
    def create_crystal_formation(self, output_dir):
        """Create crystal-like formation"""
        self._setup_plot()
        
        crystal_colors = ['#E6E6FA', '#DDA0DD', '#DA70D6', '#BA55D3', '#9370DB', '#8A2BE2']
        
        # Create multiple crystal orientations
        for i, color in enumerate(crystal_colors):
            crystal_angle = i * np.pi / 3
            crystal_scale = 1 - i * 0.1
            
            cos_c, sin_c = np.cos(crystal_angle), np.sin(crystal_angle)
            x_crystal = self.x * cos_c - self.y * sin_c
            y_crystal = self.x * sin_c + self.y * cos_c
            
            x_crystal *= crystal_scale
            y_crystal *= crystal_scale
            
            # Add crystalline structure
            plt.plot(x_crystal, y_crystal, color=color, alpha=0.8, linewidth=2)
            
            # Add crystal vertices
            vertex_indices = np.arange(0, len(x_crystal), 100)
            for j in vertex_indices:
                if j < len(x_crystal):
                    plt.scatter(x_crystal[j], y_crystal[j], c=color, s=30, 
                               marker='*', alpha=0.9)
        
        self._save_plot('dragon_crystal_formation', output_dir)
    
    def create_electric_discharge(self, output_dir):
        """Create electric discharge effect"""
        self._setup_plot()
        
        electric_colors = ['#FFFFFF', '#E0E0E0', '#00FFFF', '#80FF80', '#FFFF80']
        
        for i, color in enumerate(electric_colors):
            # Add electrical noise
            noise_strength = (5 - i) * 2
            x_electric = self.x + np.random.normal(0, noise_strength, len(self.x))
            y_electric = self.y + np.random.normal(0, noise_strength, len(self.y))
            
            # Create branching effect
            branch_prob = 0.02 * (i + 1)
            for j in range(0, len(x_electric) - 1, 10):
                if random.random() < branch_prob and j < len(x_electric) - 1:
                    # Create electrical branch
                    branch_length = random.uniform(10, 30)
                    branch_angle = random.uniform(0, 2 * np.pi)
                    
                    branch_x = x_electric[j] + branch_length * np.cos(branch_angle)
                    branch_y = y_electric[j] + branch_length * np.sin(branch_angle)
                    
                    plt.plot([x_electric[j], branch_x], [y_electric[j], branch_y], 
                            color=color, alpha=0.7, linewidth=1)
            
            # Main curve
            plt.plot(x_electric, y_electric, color=color, alpha=0.6, linewidth=1.5)
        
        self._save_plot('dragon_electric_discharge', output_dir)
    
    def create_white_background_version(self, output_dir):
        """Create bright version on white background"""
        self._setup_plot(bg_color='white')
        
        bright_colors = ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF80', '#0080FF', '#8000FF']
        
        for i, color in enumerate(bright_colors):
            offset = i * 3
            x_offset = self.x + offset
            y_offset = self.y + offset
            
            plt.plot(x_offset, y_offset, color=color, linewidth=3, alpha=0.8)
        
        self._save_plot('dragon_white_background', output_dir, bg_color='white')
    
    def create_holographic_spectrum(self, output_dir):
        """Create holographic rainbow spectrum effect"""
        self._setup_plot()
        
        # Holographic shifting colors
        holo_colors = ['#FF1493', '#00BFFF', '#32CD32', '#FFD700', '#FF4500', '#8A2BE2', '#DC143C']
        
        for i, color in enumerate(holo_colors):
            # Create holographic shift effect
            shift_x = i * 2 * np.sin(np.linspace(0, 2*np.pi, len(self.x)))
            shift_y = i * 2 * np.cos(np.linspace(0, 2*np.pi, len(self.y)))
            
            x_holo = self.x + shift_x
            y_holo = self.y + shift_y
            
            # Add iridescent shimmer
            shimmer = np.sin(np.arange(len(x_holo)) * 0.1 + i) * 0.5
            alpha_values = 0.6 + shimmer * 0.3
            
            plt.scatter(x_holo, y_holo, c=color, s=1.5, alpha=0.8)
            # Add glow effect
            plt.scatter(x_holo[::5], y_holo[::5], c=color, s=8, alpha=0.3)
        
        self._save_plot('dragon_holographic_spectrum', output_dir)
    
    def create_galaxy_nebula(self, output_dir):
        """Create galaxy nebula with cosmic colors"""
        self._setup_plot()
        
        # Cosmic nebula colors
        cosmic_colors = ['#4B0082', '#0000FF', '#00FFFF', '#FFFFFF', '#FF69B4', '#FF1493', '#8B00FF']
        
        for i, color in enumerate(cosmic_colors):
            # Create nebula-like spreading
            nebula_spread = (i + 1) * 3
            noise_x = np.random.normal(0, nebula_spread, len(self.x))
            noise_y = np.random.normal(0, nebula_spread, len(self.y))
            
            x_nebula = self.x + noise_x
            y_nebula = self.y + noise_y
            
            # Create star-like points
            star_density = 0.05 * (7 - i)  # More stars for outer layers
            star_mask = np.random.random(len(x_nebula)) < star_density
            
            # Regular nebula
            plt.scatter(x_nebula, y_nebula, c=color, s=0.8, alpha=0.6)
            # Star points
            plt.scatter(x_nebula[star_mask], y_nebula[star_mask], c='white', s=15, alpha=0.9, marker='*')
        
        self._save_plot('dragon_galaxy_nebula', output_dir)
    
    def create_fire_phoenix(self, output_dir):
        """Create phoenix rising from flames effect"""
        self._setup_plot()
        
        # Fire colors from hot to cool
        fire_colors = ['#FFFF00', '#FFD700', '#FFA500', '#FF6347', '#FF4500', '#DC143C', '#8B0000']
        
        for i, color in enumerate(fire_colors):
            # Create flame-like distortion
            flame_height = (7 - i) * 5
            flame_flicker = np.sin(self.x * 0.1 + i) * flame_height
            
            x_fire = self.x
            y_fire = self.y + flame_flicker + i * 3
            
            # Add turbulence
            turbulence_x = np.random.normal(0, (7-i) * 0.5, len(x_fire))
            turbulence_y = np.random.normal(0, (7-i) * 0.5, len(y_fire))
            
            x_fire += turbulence_x
            y_fire += turbulence_y
            
            size = 3 - i * 0.3
            alpha = 0.9 - i * 0.1
            
            plt.scatter(x_fire, y_fire, c=color, s=size, alpha=alpha)
            
            # Add ember effect
            if i < 3:
                ember_indices = np.random.choice(len(x_fire), size=len(x_fire)//20, replace=False)
                plt.scatter(x_fire[ember_indices], y_fire[ember_indices], 
                           c=color, s=12, alpha=0.8, marker='o')
        
        self._save_plot('dragon_fire_phoenix', output_dir)
    
    def create_aurora_borealis(self, output_dir):
        """Create northern lights aurora effect"""
        self._setup_plot()
        
        # Aurora colors
        aurora_colors = ['#00FF7F', '#00CED1', '#1E90FF', '#9370DB', '#FF1493', '#FFD700']
        
        for i, color in enumerate(aurora_colors):
            # Create aurora wave patterns
            wave_freq = 0.02 + i * 0.01
            wave_amp = 10 + i * 5
            
            # Vertical wave distortion (like aurora curtains)
            x_aurora = self.x + np.sin(self.y * wave_freq + i) * wave_amp
            y_aurora = self.y + np.cos(self.x * wave_freq * 0.5 + i) * (wave_amp * 0.3)
            
            # Add shimmer effect
            shimmer = np.sin(np.arange(len(x_aurora)) * 0.05 + i * 2) * 3
            x_aurora += shimmer
            
            # Create flowing lines
            plt.plot(x_aurora, y_aurora, color=color, alpha=0.7, linewidth=2)
            
            # Add glow points
            glow_indices = np.arange(0, len(x_aurora), 15)
            plt.scatter(x_aurora[glow_indices], y_aurora[glow_indices], 
                       c=color, s=20, alpha=0.6)
        
        self._save_plot('dragon_aurora_borealis', output_dir)
    
    def create_underwater_depths(self, output_dir):
        """Create underwater deep sea effect"""
        self._setup_plot()
        
        # Deep sea colors
        ocean_colors = ['#000080', '#191970', '#0000CD', '#0080FF', '#00BFFF', '#40E0D0', '#AFEEEE']
        
        for i, color in enumerate(ocean_colors):
            # Create underwater current distortion
            depth_level = i * 5
            current_strength = (7 - i) * 2
            
            # Simulate water current flow
            flow_x = np.sin(self.y * 0.03 + i) * current_strength
            flow_y = np.cos(self.x * 0.02 + i) * (current_strength * 0.5)
            
            x_ocean = self.x + flow_x
            y_ocean = self.y + flow_y - depth_level
            
            # Add bubble effect for shallow depths
            if i > 4:
                bubble_prob = 0.01
                bubble_mask = np.random.random(len(x_ocean)) < bubble_prob
                bubble_x = x_ocean[bubble_mask] + np.random.normal(0, 2, np.sum(bubble_mask))
                bubble_y = y_ocean[bubble_mask] + np.random.normal(0, 2, np.sum(bubble_mask))
                plt.scatter(bubble_x, bubble_y, c='white', s=8, alpha=0.6, marker='o')
            
            plt.scatter(x_ocean, y_ocean, c=color, s=1.5, alpha=0.8)
        
        self._save_plot('dragon_underwater_depths', output_dir)
    
    def create_metallic_chrome(self, output_dir):
        """Create metallic chrome reflection effect"""
        self._setup_plot()
        
        # Metallic chrome colors
        chrome_colors = ['#C0C0C0', '#DCDCDC', '#F5F5F5', '#FFFFFF', '#E6E6FA', '#D3D3D3', '#B0C4DE']
        
        for i, color in enumerate(chrome_colors):
            # Create metallic reflection distortion
            reflection_angle = i * np.pi / 6
            stretch_factor = 1 + i * 0.1
            
            cos_r, sin_r = np.cos(reflection_angle), np.sin(reflection_angle)
            x_chrome = self.x * cos_r - self.y * sin_r
            y_chrome = self.x * sin_r + self.y * cos_r
            
            # Apply metallic stretch
            x_chrome *= stretch_factor
            y_chrome *= (1 / stretch_factor)
            
            # Add chrome highlight streaks
            highlight_factor = np.sin(x_chrome * 0.1 + i) * 2
            y_chrome += highlight_factor
            
            plt.plot(x_chrome, y_chrome, color=color, alpha=0.8, linewidth=2)
            
            # Add reflective points
            reflect_indices = np.arange(i*20, len(x_chrome), 60)
            plt.scatter(x_chrome[reflect_indices], y_chrome[reflect_indices], 
                       c=color, s=25, alpha=0.9, marker='D')
        
        self._save_plot('dragon_metallic_chrome', output_dir)
    
    def create_volcanic_eruption(self, output_dir):
        """Create volcanic eruption with lava colors"""
        self._setup_plot()
        
        # Volcanic lava colors
        volcano_colors = ['#8B0000', '#B22222', '#DC143C', '#FF4500', '#FF6347', '#FFA500', '#FFFF00']
        
        for i, color in enumerate(volcano_colors):
            # Create volcanic explosion pattern
            explosion_center_x, explosion_center_y = np.mean(self.x), np.min(self.y)
            
            # Calculate distance from eruption point
            dist_from_center = np.sqrt((self.x - explosion_center_x)**2 + (self.y - explosion_center_y)**2)
            
            # Create upward explosive motion
            explosion_strength = (7 - i) * 0.1
            upward_force = explosion_strength * (1 / (dist_from_center + 1))
            
            x_volcano = self.x + np.random.normal(0, explosion_strength * 5, len(self.x))
            y_volcano = self.y + upward_force * 50 + np.random.normal(0, explosion_strength * 3, len(self.y))
            
            # Add lava drip effect
            drip_factor = np.sin(x_volcano * 0.05) * (7 - i)
            y_volcano -= np.abs(drip_factor) * 2
            
            plt.scatter(x_volcano, y_volcano, c=color, s=2 + i * 0.5, alpha=0.8)
            
            # Add lava splatter
            if i < 4:
                splatter_indices = np.random.choice(len(x_volcano), size=len(x_volcano)//30, replace=False)
                splatter_x = x_volcano[splatter_indices] + np.random.normal(0, 10, len(splatter_indices))
                splatter_y = y_volcano[splatter_indices] + np.random.normal(0, 10, len(splatter_indices))
                plt.scatter(splatter_x, splatter_y, c=color, s=15, alpha=0.7)
        
        self._save_plot('dragon_volcanic_eruption', output_dir)
    
    def create_cyberpunk_neon(self, output_dir):
        """Create cyberpunk neon city effect"""
        self._setup_plot()
        
        # Cyberpunk neon colors
        cyber_colors = ['#FF0080', '#00FFFF', '#FFFF00', '#FF1493', '#00FF41', '#8A2BE2', '#FF4500']
        
        for i, color in enumerate(cyber_colors):
            # Create digital glitch effect
            glitch_intensity = (i % 3) * 2
            digital_noise_x = np.random.choice([-glitch_intensity, 0, glitch_intensity], len=self.x)
            digital_noise_y = np.random.choice([-glitch_intensity, 0, glitch_intensity], len=self.y)
            
            x_cyber = self.x + digital_noise_x
            y_cyber = self.y + digital_noise_y
            
            # Add neon glow lines
            for j in range(0, len(x_cyber)-1, 20):
                if j+1 < len(x_cyber):
                    plt.plot([x_cyber[j], x_cyber[j+1]], [y_cyber[j], y_cyber[j+1]], 
                            color=color, alpha=0.8, linewidth=3)
                    # Outer glow
                    plt.plot([x_cyber[j], x_cyber[j+1]], [y_cyber[j], y_cyber[j+1]], 
                            color=color, alpha=0.3, linewidth=8)
            
            # Add neon nodes
            node_indices = np.arange(i*10, len(x_cyber), 50)
            plt.scatter(x_cyber[node_indices], y_cyber[node_indices], 
                       c=color, s=30, alpha=0.9, marker='s')
        
        self._save_plot('dragon_cyberpunk_neon', output_dir)
    
    def create_ultra_detailed_microscope(self, output_dir):
        """Create ultra-detailed microscopic view with fine structures"""
        self._setup_plot()
        
        # Microscopic detail colors
        micro_colors = ['#00FFFF', '#40E0D0', '#20B2AA', '#008B8B', '#006666']
        
        for i, color in enumerate(micro_colors):
            # Create different magnification levels
            magnification = 1 + i * 0.05
            detail_offset = i * 0.1
            
            # Add microscopic noise for texture
            micro_x = self.x * magnification + np.random.normal(0, 0.05, len(self.x))
            micro_y = self.y * magnification + np.random.normal(0, 0.05, len(self.y))
            
            # Create variable point sizes for depth perception
            distances = np.sqrt(micro_x**2 + micro_y**2)
            point_sizes = 0.2 + (distances / np.max(distances)) * 2
            
            plt.scatter(micro_x, micro_y, s=point_sizes, c=color, alpha=0.7)
            
            # Add fine connecting lines
            for j in range(0, len(micro_x)-1, 200):
                if j+1 < len(micro_x):
                    plt.plot([micro_x[j], micro_x[j+1]], [micro_y[j], micro_y[j+1]], 
                            color=color, alpha=0.3, linewidth=0.2)
        
        self._save_plot('dragon_ultra_detailed_microscope', output_dir)
    
    def create_fractal_zoom_details(self, output_dir):
        """Create fractal zoom showing self-similar details at different scales"""
        self._setup_plot()
        
        # Zoom level colors
        zoom_colors = ['#FF6B6B', '#FFE66D', '#95E1D3', '#A8E6CF', '#C7CEEA']
        
        center_x, center_y = np.mean(self.x), np.mean(self.y)
        
        for i, color in enumerate(zoom_colors):
            # Create progressive zoom levels
            zoom_factor = 1 + i * 0.8
            focus_radius = 3.0 / (1 + i * 0.5)
            
            # Select points within focus area
            distances = np.sqrt((self.x - center_x)**2 + (self.y - center_y)**2)
            mask = distances < focus_radius
            
            if np.any(mask):
                x_zoom = self.x[mask]
                y_zoom = self.y[mask]
                
                # Apply zoom transformation
                x_focused = (x_zoom - center_x) * zoom_factor + center_x
                y_focused = (y_zoom - center_y) * zoom_factor + center_y
                
                # Add fractal detail enhancement
                detail_freq = 10 + i * 5
                x_enhanced = x_focused + np.sin(y_focused * detail_freq) * (0.1 / zoom_factor)
                y_enhanced = y_focused + np.cos(x_focused * detail_freq) * (0.1 / zoom_factor)
                
                # Scale point sizes with zoom
                sizes = (0.5 + i * 0.3) * np.ones(len(x_enhanced))
                
                plt.scatter(x_enhanced, y_enhanced, s=sizes, c=color, alpha=0.8)
                
                # Add detail boundary
                if i > 0:
                    circle = plt.Circle((center_x, center_y), focus_radius * zoom_factor, 
                                      fill=False, color=color, alpha=0.5, linewidth=1)
                    plt.gca().add_patch(circle)
        
        self._save_plot('dragon_fractal_zoom_details', output_dir)
    
    def create_fiber_optic_network(self, output_dir):
        """Create fiber optic network showing internal light transmission"""
        self._setup_plot()
        
        # Fiber optic colors - bright cores with halos
        fiber_colors = ['#FFFFFF', '#FFFF99', '#99FFFF', '#FF99FF', '#99FF99']
        
        for i, color in enumerate(fiber_colors):
            # Create fiber bundle effect
            bundle_offset = i * 0.3
            fiber_thickness = 0.2 + i * 0.1
            
            # Add fiber wave guides
            x_fiber = self.x + np.sin(self.y * 2 + i) * fiber_thickness
            y_fiber = self.y + np.cos(self.x * 2 + i) * fiber_thickness
            
            # Create core and cladding layers
            core_size = 0.3 + i * 0.1
            cladding_size = core_size * 3
            
            # Draw fiber cladding (outer layer)
            plt.scatter(x_fiber, y_fiber, s=cladding_size, c=color, alpha=0.2)
            
            # Draw fiber core (inner light)
            plt.scatter(x_fiber, y_fiber, s=core_size, c=color, alpha=0.9)
            
            # Add light pulse effects
            pulse_indices = np.arange(i*20, len(x_fiber), 80)
            if len(pulse_indices) > 0:
                plt.scatter(x_fiber[pulse_indices], y_fiber[pulse_indices], 
                           s=core_size*4, c='white', alpha=0.8, marker='*')
        
        self._save_plot('dragon_fiber_optic_network', output_dir)
    
    def create_crystalline_lattice_details(self, output_dir):
        """Create detailed crystalline lattice structure"""
        self._setup_plot()
        
        # Crystal lattice colors
        lattice_colors = ['#E6E6FA', '#D8BFD8', '#DDA0DD', '#DA70D6', '#BA55D3']
        
        for i, color in enumerate(lattice_colors):
            # Create lattice grid
            lattice_size = 0.5 + i * 0.2
            
            # Snap points to lattice grid
            x_lattice = np.round(self.x / lattice_size) * lattice_size
            y_lattice = np.round(self.y / lattice_size) * lattice_size
            
            # Add crystalline imperfections
            imperfection = 0.1 / (i + 1)
            x_crystal = x_lattice + np.random.normal(0, imperfection, len(x_lattice))
            y_crystal = y_lattice + np.random.normal(0, imperfection, len(y_lattice))
            
            # Create crystal facets
            facet_size = 1.5 + i * 0.5
            plt.scatter(x_crystal, y_crystal, s=facet_size, c=color, alpha=0.8, marker='D')
            
            # Add crystal bonds
            bond_threshold = lattice_size * 1.5
            for j in range(0, len(x_crystal)-1, 50):
                for k in range(j+1, min(j+10, len(x_crystal))):
                    distance = np.sqrt((x_crystal[k] - x_crystal[j])**2 + 
                                     (y_crystal[k] - y_crystal[j])**2)
                    if distance < bond_threshold:
                        bond_strength = 1 - (distance / bond_threshold)
                        plt.plot([x_crystal[j], x_crystal[k]], [y_crystal[j], y_crystal[k]], 
                                color=color, alpha=bond_strength*0.5, linewidth=0.8)
        
        self._save_plot('dragon_crystalline_lattice_details', output_dir)
    
    def create_quantum_interference_patterns(self, output_dir):
        """Create quantum interference patterns with wave details"""
        self._setup_plot()
        
        # Quantum state colors
        quantum_colors = ['#FF0066', '#FF3366', '#FF6666', '#FF9966', '#FFCC66']
        
        for i, color in enumerate(quantum_colors):
            # Create quantum wave functions
            wavelength = 0.3 + i * 0.1
            phase_offset = i * np.pi / 3
            
            # Quantum wave interference
            wave1 = np.sin(self.x / wavelength + phase_offset)
            wave2 = np.cos(self.y / wavelength + phase_offset)
            interference = wave1 * wave2
            
            # Probability density (square of wave function)
            probability = interference**2
            
            # Create quantum dots at high probability regions
            quantum_threshold = 0.5 - i * 0.1
            quantum_mask = probability > quantum_threshold
            
            if np.any(quantum_mask):
                x_quantum = self.x[quantum_mask]
                y_quantum = self.y[quantum_mask]
                prob_values = probability[quantum_mask]
                
                # Dot sizes based on probability
                dot_sizes = 0.5 + prob_values * 3
                
                plt.scatter(x_quantum, y_quantum, s=dot_sizes, c=color, alpha=0.9)
                
                # Add uncertainty principle visualization
                uncertainty_x = np.random.normal(0, wavelength*0.1, len(x_quantum))
                uncertainty_y = np.random.normal(0, wavelength*0.1, len(y_quantum))
                
                x_uncertain = x_quantum + uncertainty_x
                y_uncertain = y_quantum + uncertainty_y
                
                plt.scatter(x_uncertain, y_uncertain, s=dot_sizes*0.3, 
                           c=color, alpha=0.3, marker='.')
                
                # Add wave function lines
                for j in range(0, len(x_quantum)-1, 30):
                    if j+1 < len(x_quantum):
                        # Phase relationship
                        phase_diff = np.sin((x_quantum[j] + y_quantum[j]) / wavelength)
                        if phase_diff > 0:
                            plt.plot([x_quantum[j], x_quantum[j+1]], 
                                   [y_quantum[j], y_quantum[j+1]], 
                                   color=color, alpha=0.5, linewidth=0.5, linestyle='--')
        
        self._save_plot('dragon_quantum_interference_patterns', output_dir)
    
    def create_minimalist_bw(self, output_dir):
        """Create minimalist black and white version"""
        self._setup_plot(bg_color='white')
        
        # Simple black lines on white background
        plt.plot(self.x, self.y, color='black', linewidth=1.5, alpha=0.8)
        
        # Add subtle dots at key points
        highlight_indices = np.arange(0, len(self.x), 100)
        plt.scatter(self.x[highlight_indices], self.y[highlight_indices], 
                   c='black', s=3, alpha=0.6)
        
        self._save_plot('dragon_minimalist_bw', output_dir, bg_color='white')
    
    def create_sketch_style_bw(self, output_dir):
        """Create hand-drawn sketch style in black and white"""
        self._setup_plot(bg_color='white')
        
        # Create sketch-like lines with varying thickness
        for i in range(0, len(self.x)-1, 5):
            if i+1 < len(self.x):
                # Add slight randomness for hand-drawn effect
                x_sketch = [self.x[i] + np.random.normal(0, 0.02), 
                           self.x[i+1] + np.random.normal(0, 0.02)]
                y_sketch = [self.y[i] + np.random.normal(0, 0.02), 
                           self.y[i+1] + np.random.normal(0, 0.02)]
                
                # Varying line thickness
                thickness = np.random.uniform(0.5, 2.0)
                alpha = np.random.uniform(0.3, 0.9)
                
                plt.plot(x_sketch, y_sketch, color='black', 
                        linewidth=thickness, alpha=alpha)
        
        self._save_plot('dragon_sketch_style_bw', output_dir, bg_color='white')
    
    def create_high_contrast_bw(self, output_dir):
        """Create high contrast black and white version"""
        self._setup_plot()
        
        # White lines on black background with high contrast
        plt.plot(self.x, self.y, color='white', linewidth=2, alpha=1.0)
        
        # Add white glow effect
        plt.plot(self.x, self.y, color='white', linewidth=6, alpha=0.3)
        
        # Bright white accent points
        accent_indices = np.arange(0, len(self.x), 50)
        plt.scatter(self.x[accent_indices], self.y[accent_indices], 
                   c='white', s=8, alpha=1.0)
        
        self._save_plot('dragon_high_contrast_bw', output_dir)
    
    def create_crosshatch_bw(self, output_dir):
        """Create crosshatch style black and white version"""
        self._setup_plot(bg_color='white')
        
        # Main curve
        plt.plot(self.x, self.y, color='black', linewidth=1, alpha=0.8)
        
        # Add crosshatch patterns
        crosshatch_spacing = 0.2
        
        for i in range(0, len(self.x)-1, 20):
            if i+1 < len(self.x):
                # Calculate perpendicular direction
                dx = self.x[i+1] - self.x[i]
                dy = self.y[i+1] - self.y[i]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Perpendicular unit vector
                    perp_x = -dy / length
                    perp_y = dx / length
                    
                    # Draw crosshatch lines
                    for j in range(-2, 3):
                        start_x = self.x[i] + j * crosshatch_spacing * perp_x
                        start_y = self.y[i] + j * crosshatch_spacing * perp_y
                        end_x = start_x + dx * 0.3
                        end_y = start_y + dy * 0.3
                        
                        plt.plot([start_x, end_x], [start_y, end_y], 
                                color='black', linewidth=0.5, alpha=0.4)
        
        self._save_plot('dragon_crosshatch_bw', output_dir, bg_color='white')
    
    def create_ink_wash_bw(self, output_dir):
        """Create ink wash style black and white version"""
        self._setup_plot(bg_color='white')
        
        # Create multiple layers with different transparencies
        wash_layers = [
            (1.0, 0.1, 8),    # Light wash, large brush
            (0.8, 0.3, 4),    # Medium wash
            (0.6, 0.6, 2),    # Dark wash
            (1.0, 1.0, 1),    # Ink line
        ]
        
        for scale, alpha, linewidth in wash_layers:
            # Add slight offset for wash effect
            offset_x = np.random.normal(0, 0.05, len(self.x))
            offset_y = np.random.normal(0, 0.05, len(self.y))
            
            x_wash = self.x + offset_x
            y_wash = self.y + offset_y
            
            plt.plot(x_wash, y_wash, color='black', 
                    linewidth=linewidth, alpha=alpha)
        
        self._save_plot('dragon_ink_wash_bw', output_dir, bg_color='white')
    
    def create_geometric_bw(self, output_dir):
        """Create geometric black and white pattern"""
        self._setup_plot()
        
        # Create geometric shapes along the curve
        for i in range(0, len(self.x), 30):
            size = abs(np.sin(i * 0.1)) * 10 + 2
            
            # Alternate between different geometric shapes
            shape_type = i % 4
            
            if shape_type == 0:
                # Square
                square = plt.Rectangle((self.x[i] - size/2, self.y[i] - size/2), 
                                     size, size, fill=False, 
                                     edgecolor='white', linewidth=1)
                plt.gca().add_patch(square)
            elif shape_type == 1:
                # Circle
                circle = plt.Circle((self.x[i], self.y[i]), size/2, 
                                  fill=False, edgecolor='white', linewidth=1)
                plt.gca().add_patch(circle)
            elif shape_type == 2:
                # Triangle
                triangle_x = [self.x[i], self.x[i] - size/2, self.x[i] + size/2, self.x[i]]
                triangle_y = [self.y[i] + size/2, self.y[i] - size/2, 
                             self.y[i] - size/2, self.y[i] + size/2]
                plt.plot(triangle_x, triangle_y, color='white', linewidth=1)
            else:
                # Diamond
                diamond_x = [self.x[i], self.x[i] - size/2, self.x[i], 
                            self.x[i] + size/2, self.x[i]]
                diamond_y = [self.y[i] + size/2, self.y[i], self.y[i] - size/2, 
                            self.y[i], self.y[i] + size/2]
                plt.plot(diamond_x, diamond_y, color='white', linewidth=1)
        
        self._save_plot('dragon_geometric_bw', output_dir)
    
    def create_negative_space_bw(self, output_dir):
        """Create negative space black and white version"""
        self._setup_plot()
        
        # Fill background area around the curve
        from matplotlib.patches import Polygon
        
        # Create thick white border around the curve
        plt.plot(self.x, self.y, color='white', linewidth=15, alpha=1.0)
        
        # Add the main curve in thinner line
        plt.plot(self.x, self.y, color='black', linewidth=2, alpha=1.0)
        
        # Add negative space elements
        for i in range(0, len(self.x), 80):
            # Create small black filled circles as negative space
            circle = plt.Circle((self.x[i], self.y[i]), 0.3, 
                              fill=True, facecolor='black', alpha=0.8)
            plt.gca().add_patch(circle)
        
        self._save_plot('dragon_negative_space_bw', output_dir)
    
    def create_woodcut_bw(self, output_dir):
        """Create woodcut/linocut style black and white version"""
        self._setup_plot(bg_color='white')
        
        # Bold, chunky lines like woodcut
        plt.plot(self.x, self.y, color='black', linewidth=6, alpha=1.0)
        
        # Add texture lines perpendicular to curve
        for i in range(0, len(self.x)-1, 15):
            if i+1 < len(self.x):
                # Direction vector
                dx = self.x[i+1] - self.x[i]
                dy = self.y[i+1] - self.y[i]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Perpendicular vector
                    perp_x = -dy / length * 2
                    perp_y = dx / length * 2
                    
                    # Draw texture lines
                    for j in range(-1, 2):
                        start_x = self.x[i] + j * perp_x
                        start_y = self.y[i] + j * perp_y
                        end_x = start_x + perp_x * 0.5
                        end_y = start_y + perp_y * 0.5
                        
                        plt.plot([start_x, end_x], [start_y, end_y], 
                                color='black', linewidth=2, alpha=0.6)
        
        self._save_plot('dragon_woodcut_bw', output_dir, bg_color='white')
    
    def create_etching_bw(self, output_dir):
        """Create etching style with fine parallel lines"""
        self._setup_plot(bg_color='white')
        
        # Main curve outline
        plt.plot(self.x, self.y, color='black', linewidth=1, alpha=0.9)
        
        # Create parallel hatching lines
        hatch_spacing = 0.15
        hatch_angle = np.pi / 4  # 45 degree angle
        
        # Find bounding box
        x_min, x_max = np.min(self.x) - 5, np.max(self.x) + 5
        y_min, y_max = np.min(self.y) - 5, np.max(self.y) + 5
        
        # Create parallel lines across entire area
        for offset in np.arange(-20, 20, hatch_spacing):
            x_start = x_min + offset * np.cos(hatch_angle + np.pi/2)
            y_start = y_min + offset * np.sin(hatch_angle + np.pi/2)
            x_end = x_max + offset * np.cos(hatch_angle + np.pi/2)
            y_end = y_max + offset * np.sin(hatch_angle + np.pi/2)
            
            plt.plot([x_start, x_end], [y_start, y_end], 
                    color='black', linewidth=0.3, alpha=0.4)
        
        self._save_plot('dragon_etching_bw', output_dir, bg_color='white')
    
    def create_stencil_bw(self, output_dir):
        """Create stencil style with bold cutouts"""
        self._setup_plot(bg_color='white')
        
        # Thick black outline
        plt.plot(self.x, self.y, color='black', linewidth=8, alpha=1.0)
        
        # Add stencil bridges (gaps in the line)
        bridge_spacing = 50
        for i in range(0, len(self.x), bridge_spacing):
            if i + 10 < len(self.x):
                # Create white gap (bridge)
                plt.plot(self.x[i:i+10], self.y[i:i+10], 
                        color='white', linewidth=12, alpha=1.0)
        
        # Add stencil corner markers
        corners = [0, len(self.x)//4, len(self.x)//2, 3*len(self.x)//4, -1]
        for corner in corners:
            if corner < len(self.x):
                # Small registration marks
                plt.scatter(self.x[corner], self.y[corner], 
                           c='black', s=20, marker='+', linewidth=2)
        
        self._save_plot('dragon_stencil_bw', output_dir, bg_color='white')
    
    def create_calligraphy_bw(self, output_dir):
        """Create calligraphy brush stroke style"""
        self._setup_plot(bg_color='white')
        
        # Variable width brush strokes
        for i in range(0, len(self.x)-1, 3):
            if i+1 < len(self.x):
                # Calculate speed/curvature for brush width
                if i > 0 and i < len(self.x) - 2:
                    curvature = abs((self.x[i+1] - self.x[i-1]) * (self.y[i+2] - self.y[i]) - 
                                   (self.y[i+1] - self.y[i-1]) * (self.x[i+2] - self.x[i]))
                    brush_width = 0.5 + curvature * 10
                    brush_width = min(brush_width, 4.0)  # Cap maximum width
                else:
                    brush_width = 1.0
                
                # Add brush texture
                alpha = 0.7 + np.random.normal(0, 0.1)
                alpha = max(0.3, min(1.0, alpha))
                
                plt.plot([self.x[i], self.x[i+1]], [self.y[i], self.y[i+1]], 
                        color='black', linewidth=brush_width, alpha=alpha)
        
        # Add ink splatters
        for _ in range(20):
            splatter_x = np.random.choice(self.x)
            splatter_y = np.random.choice(self.y)
            splatter_x += np.random.normal(0, 1)
            splatter_y += np.random.normal(0, 1)
            splatter_size = np.random.uniform(1, 8)
            
            plt.scatter(splatter_x, splatter_y, c='black', 
                       s=splatter_size, alpha=0.6)
        
        self._save_plot('dragon_calligraphy_bw', output_dir, bg_color='white')
    
    def create_silhouette_bw(self, output_dir):
        """Create filled silhouette version"""
        self._setup_plot()
        
        # Create filled polygon from curve
        plt.fill(self.x, self.y, color='white', alpha=1.0)
        plt.plot(self.x, self.y, color='white', linewidth=2, alpha=1.0)
        
        # Add dramatic lighting effect
        center_x, center_y = np.mean(self.x), np.mean(self.y)
        
        # Create gradient effect with circles
        for radius in np.arange(1, 20, 0.5):
            circle = plt.Circle((center_x + 5, center_y + 10), radius, 
                               fill=False, edgecolor='white', 
                               alpha=0.1, linewidth=0.5)
            plt.gca().add_patch(circle)
        
        self._save_plot('dragon_silhouette_bw', output_dir)
    
    def create_mosaic_bw(self, output_dir):
        """Create mosaic tile effect in black and white"""
        self._setup_plot(bg_color='white')
        
        # Create mosaic tiles along the curve
        tile_size = 0.3
        for i in range(0, len(self.x), 8):
            # Determine tile color based on position
            tile_color = 'black' if (i // 8) % 2 == 0 else 'white'
            edge_color = 'white' if tile_color == 'black' else 'black'
            
            # Create square tile
            square = plt.Rectangle((self.x[i] - tile_size/2, self.y[i] - tile_size/2), 
                                 tile_size, tile_size, 
                                 facecolor=tile_color, edgecolor=edge_color, 
                                 linewidth=0.5, alpha=0.9)
            plt.gca().add_patch(square)
        
        # Add grout lines
        plt.plot(self.x, self.y, color='gray', linewidth=1, alpha=0.5)
        
        self._save_plot('dragon_mosaic_bw', output_dir, bg_color='white')
    
    def create_blueprint_bw(self, output_dir):
        """Create technical blueprint style"""
        self._setup_plot(bg_color='#1a237e')  # Dark blue background
        
        # White technical lines
        plt.plot(self.x, self.y, color='white', linewidth=1, alpha=0.9)
        
        # Add grid lines
        x_min, x_max = np.min(self.x) - 5, np.max(self.x) + 5
        y_min, y_max = np.min(self.y) - 5, np.max(self.y) + 5
        
        # Vertical grid lines
        for x in np.arange(x_min, x_max, 2):
            plt.axvline(x, color='cyan', alpha=0.3, linewidth=0.5)
        
        # Horizontal grid lines
        for y in np.arange(y_min, y_max, 2):
            plt.axhline(y, color='cyan', alpha=0.3, linewidth=0.5)
        
        # Add dimension lines
        for i in range(0, len(self.x), 100):
            if i + 50 < len(self.x):
                # Dimension arrow
                plt.annotate('', xy=(self.x[i+50], self.y[i+50]), 
                           xytext=(self.x[i], self.y[i]),
                           arrowprops=dict(arrowstyle='<->', color='yellow', lw=0.8))
        
        # Add technical annotations
        plt.text(np.mean(self.x), np.max(self.y) + 2, 'DRAGON CURVE', 
                color='white', fontsize=12, ha='center', weight='bold')
        plt.text(np.min(self.x), np.min(self.y) - 2, 'SCALE: 1:1', 
                color='white', fontsize=8)
        
        self._save_plot('dragon_blueprint_bw', output_dir, bg_color='#1a237e')

def main():
    # Create project directory and a single output folder for this run (no nested 'dragon_images' folder)
    project_dir = Path(r'c:\Users\Marlis\OneDrive\Dokumente\Projekte\creative-coding\dragon_curve_variations')
    run_dir = project_dir / datetime.now().strftime('run_%Y%m%d_%H%M%S')
    
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Dragon Curve Art variations...")
    
    # Create dragon curve art with 13 iterations for good detail
    dragon_art = DragonCurveArt(iterations=13, seed=42)
    
    variations = [
        ('Rainbow Gradient', dragon_art.create_rainbow_gradient),
        ('Neon Segments', dragon_art.create_neon_segments),
        ('Spiral View', dragon_art.create_spiral_view),
        ('Fractal Explosion', dragon_art.create_fractal_explosion),
        ('Wave Distortion', dragon_art.create_wave_distortion),
        ('Psychedelic Layers', dragon_art.create_psychedelic_layers),
        ('Psychedelic Layers White BG', dragon_art.create_psychedelic_layers_white_bg),
        ('Crystal Formation', dragon_art.create_crystal_formation),
        ('Electric Discharge', dragon_art.create_electric_discharge),
        ('White Background', dragon_art.create_white_background_version),
        ('Holographic Spectrum', dragon_art.create_holographic_spectrum),
        ('Galaxy Nebula', dragon_art.create_galaxy_nebula),
        ('Fire Phoenix', dragon_art.create_fire_phoenix),
        ('Aurora Borealis', dragon_art.create_aurora_borealis),
        ('Underwater Depths', dragon_art.create_underwater_depths),
        ('Metallic Chrome', dragon_art.create_metallic_chrome),
        ('Volcanic Eruption', dragon_art.create_volcanic_eruption),
        ('Cyberpunk Neon', dragon_art.create_cyberpunk_neon),
        ('Ultra Detailed Microscope', dragon_art.create_ultra_detailed_microscope),
        ('Fractal Zoom Details', dragon_art.create_fractal_zoom_details),
        ('Fiber Optic Network', dragon_art.create_fiber_optic_network),
        ('Crystalline Lattice Details', dragon_art.create_crystalline_lattice_details),
        ('Quantum Interference Patterns', dragon_art.create_quantum_interference_patterns),
        ('Minimalist BW', dragon_art.create_minimalist_bw),
        ('Sketch Style BW', dragon_art.create_sketch_style_bw),
        ('High Contrast BW', dragon_art.create_high_contrast_bw),
        # 'Stippled BW' removed because it takes too long to generate
        ('Crosshatch BW', dragon_art.create_crosshatch_bw),
        ('Ink Wash BW', dragon_art.create_ink_wash_bw),
        ('Geometric BW', dragon_art.create_geometric_bw),
        ('Negative Space BW', dragon_art.create_negative_space_bw),
        ('Woodcut BW', dragon_art.create_woodcut_bw),
        ('Etching BW', dragon_art.create_etching_bw),
        ('Stencil BW', dragon_art.create_stencil_bw),
        ('Calligraphy BW', dragon_art.create_calligraphy_bw),
        ('Silhouette BW', dragon_art.create_silhouette_bw),
        ('Mosaic BW', dragon_art.create_mosaic_bw),
        ('Blueprint BW', dragon_art.create_blueprint_bw),
    ]
    
    for name, method in variations:
        print(f"Creating {name}...")
        try:
            method(run_dir)
            print(f"✓ Saved to: {run_dir}")
        except Exception as e:
            print(f"✗ Error creating {name}: {e}")
    
    print(f"\nAll Dragon Curve Art saved to: {run_dir}")
    print("Generated variations:")
    for name, _ in variations:
        print(f"- {name}")

if __name__ == "__main__":
    main()
