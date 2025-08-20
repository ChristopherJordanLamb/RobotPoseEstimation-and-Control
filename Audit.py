import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from collections import defaultdict

class GestureVisualizer:
    def __init__(self):
        self.data_folder = r"C:\Users\Christopher\RobotPoseEstimation and Control\gesture_training_data"
        self.gesture_files = {}  # {category: [file_paths]}
        self.current_file_data = None
        self.current_category = None
        self.current_file = None
        self.current_frame_idx = 0
        self.playing = False
        self.animation = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Gesture Training Data Visualizer")
        self.root.geometry("1400x900")
        
        self.setup_ui()
        self.load_gesture_files()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File management
        file_frame = ttk.LabelFrame(control_frame, text="Gesture Files", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Category selection
        ttk.Label(file_frame, text="Category:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(file_frame, textvariable=self.category_var, state="readonly", width=20)
        self.category_combo.grid(row=0, column=1, padx=(0, 10), sticky=tk.W)
        self.category_combo.bind('<<ComboboxSelected>>', self.on_category_select)
        
        # File selection
        ttk.Label(file_frame, text="File:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(file_frame, textvariable=self.file_var, state="readonly", width=35)
        self.file_combo.grid(row=0, column=3, padx=(0, 10), sticky=tk.W)
        self.file_combo.bind('<<ComboboxSelected>>', self.on_file_select)
        
        # Reload button
        ttk.Button(file_frame, text="Reload Files", command=self.load_gesture_files).grid(row=0, column=4, padx=(10, 0))
        
        # Playback controls
        playback_frame = ttk.LabelFrame(control_frame, text="Playback", padding=10)
        playback_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self.toggle_playback, state="disabled")
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_button = ttk.Button(playback_frame, text="Reset", command=self.reset_animation, state="disabled")
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Frame control
        ttk.Label(playback_frame, text="Frame:").pack(side=tk.LEFT, padx=(10, 5))
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(playback_frame, variable=self.frame_var, orient=tk.HORIZONTAL, 
                                   length=300, command=self.on_frame_change, state="disabled")
        self.frame_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.frame_label = ttk.Label(playback_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Speed control
        ttk.Label(playback_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(playback_frame, variable=self.speed_var, orient=tk.HORIZONTAL, 
                               length=100, from_=0.1, to=3.0)
        speed_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.speed_label = ttk.Label(playback_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT)
        self.speed_var.trace('w', self.on_speed_change)
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Gesture Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Visualization area
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(14, 8))
        
        # Create subplots
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_distance = self.fig.add_subplot(222)
        self.ax_motion = self.fig.add_subplot(223)
        self.ax_measurements = self.fig.add_subplot(224)
        
        self.fig.tight_layout()
        
        # Embed matplotlib in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_gesture_files(self):
        """Load and categorize all JSON files from the gesture training data folder"""
        self.gesture_files = defaultdict(list)
        
        if not os.path.exists(self.data_folder):
            messagebox.showerror("Error", f"Folder not found: {self.data_folder}")
            return
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.data_folder, "*.json"))
        
        if not json_files:
            messagebox.showwarning("Warning", f"No JSON files found in: {self.data_folder}")
            return
        
        # Categorize files by gesture name (everything before the date)
        for file_path in json_files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Split by underscore and find where the date starts (8 digit number)
            parts = name_without_ext.split('_')
            category_parts = []
            
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 8:  # Date part (YYYYMMDD)
                    break
                category_parts.append(part)
            
            if category_parts:
                category = '_'.join(category_parts)
                self.gesture_files[category].append(file_path)
        
        # Sort files within each category by date/time
        for category in self.gesture_files:
            self.gesture_files[category].sort(key=lambda x: os.path.basename(x))
        
        # Update UI
        categories = sorted(self.gesture_files.keys())
        self.category_combo['values'] = categories
        
        if categories:
            self.category_combo.set(categories[0])
            self.on_category_select()
        
        total_files = sum(len(files) for files in self.gesture_files.values())
        print(f"Loaded {total_files} gesture files in {len(categories)} categories")
    
    def on_speed_change(self, *args):
        speed = self.speed_var.get()
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def on_category_select(self, event=None):
        """Update file dropdown when category is selected"""
        if not self.category_var.get():
            return
        
        category = self.category_var.get()
        files = self.gesture_files[category]
        
        # Show just filenames for easier reading
        file_display_names = [os.path.basename(f) for f in files]
        self.file_combo['values'] = file_display_names
        
        if file_display_names:
            self.file_combo.set(file_display_names[0])
            self.on_file_select()
    
    def on_file_select(self, event=None):
        """Load selected file and update visualization"""
        if not self.category_var.get() or not self.file_var.get():
            return
        
        category = self.category_var.get()
        filename = self.file_var.get()
        
        # Find full path
        full_path = None
        for path in self.gesture_files[category]:
            if os.path.basename(path) == filename:
                full_path = path
                break
        
        if not full_path:
            messagebox.showerror("Error", f"File not found: {filename}")
            return
        
        try:
            with open(full_path, 'r') as f:
                self.current_file_data = json.load(f)
            
            self.current_category = category
            self.current_file = filename
            
            # Reset playback
            self.playing = False
            self.current_frame_idx = 0
            self.play_button.config(text="Play", state="normal")
            self.reset_button.config(state="normal")
            
            # Update frame scale
            if 'sequence' in self.current_file_data:
                num_frames = len(self.current_file_data['sequence'])
                self.frame_scale.config(to=num_frames-1 if num_frames > 0 else 0, state="normal")
                self.frame_var.set(0)
            
            # Update info and visualization
            self.update_info_panel()
            self.visualize_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file {filename}: {str(e)}")
    
    def update_info_panel(self):
        """Update the information panel with current file details"""
        if not self.current_file_data:
            return
        
        info = f"Category: {self.current_category}\n"
        info += f"File: {self.current_file}\n"
        info += f"Label: {self.current_file_data.get('label', 'Unknown')}\n"
        
        if 'sequence' in self.current_file_data:
            frames = self.current_file_data['sequence']
            info += f"Total Frames: {len(frames)} ({self.current_file_data.get('duration', 'Unknown')}s)\n"
            
            if frames:
                # Hand analysis
                hand_counts = defaultdict(int)
                max_motion = 0
                
                for frame in frames:
                    if 'hands' in frame:
                        for hand in frame['hands']:
                            hand_label = hand.get('hand_label', 'Unknown')
                            hand_counts[hand_label] += 1
                            
                            if 'motion' in hand and 'overall_motion_magnitude' in hand['motion']:
                                max_motion = max(max_motion, hand['motion']['overall_motion_magnitude'])
                
                if hand_counts:
                    hands_info = ', '.join([f"{label} ({count} frames)" for label, count in hand_counts.items()])
                    info += f"Hands: {hands_info}\n"
                
                info += f"Max motion: {max_motion:.2f} cm/s"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def on_frame_change(self, value):
        if not self.playing:
            self.current_frame_idx = int(float(value))
            self.visualize_frame()
    
    def visualize_frame(self):
        if not self.current_file_data or 'sequence' not in self.current_file_data:
            return
        
        frames = self.current_file_data['sequence']
        
        if not frames or self.current_frame_idx >= len(frames):
            return
        
        frame_data = frames[self.current_frame_idx]
        
        # Clear all plots
        self.ax_3d.clear()
        self.ax_distance.clear()
        self.ax_motion.clear()
        self.ax_measurements.clear()
        
        # 3D Hand positions
        self.ax_3d.set_title(f"Hand Positions - Frame {self.current_frame_idx}")
        colors = ['red', 'blue']
        
        if 'hands' in frame_data:
            for i, hand in enumerate(frame_data['hands']):
                # Plot hand center
                distance = hand.get('distance', 0)
                hand_label = hand.get('hand_label', f'Hand {i}')
                
                # Approximate 3D position (you may need to adjust based on your coordinate system)
                x, y, z = 0, 0, distance  # Simplified - you might have actual 3D coords
                
                self.ax_3d.scatter([x], [y], [z], c=colors[i % len(colors)], s=100, 
                                 label=f"{hand_label} ({distance:.1f}cm)")
                
                # Plot fingertips if available
                if 'fingertip_positions' in hand:
                    fingertips = hand['fingertip_positions']
                    for finger_name, pos in fingertips.items():
                        if len(pos) >= 3:
                            self.ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], 
                                             c=colors[i % len(colors)], s=30, alpha=0.7)
        
        self.ax_3d.set_xlabel('X (cm)')
        self.ax_3d.set_ylabel('Y (cm)')
        self.ax_3d.set_zlabel('Z (cm)')
        self.ax_3d.legend()
        
        # Distance over time
        self.plot_distance_timeline(frames, self.current_frame_idx)
        
        # Motion magnitude over time
        self.plot_motion_timeline(frames, self.current_frame_idx)
        
        # Palm measurements
        self.plot_palm_measurements(frame_data)
        
        # Update frame label
        self.frame_label.config(text=f"{self.current_frame_idx}/{len(frames)-1}")
        
        self.canvas.draw()
    
    def plot_distance_timeline(self, frames, current_frame):
        if not frames:
            return
        
        timestamps = [frame['timestamp'] - frames[0]['timestamp'] for frame in frames]
        
        hand_distances = defaultdict(list)
        for frame in frames:
            if 'hands' in frame:
                for hand in frame['hands']:
                    label = hand.get('hand_label', 'Unknown')
                    distance = hand.get('distance', 0)
                    hand_distances[label].append(distance)
            else:
                # Pad with None for frames without hands
                for label in hand_distances.keys():
                    hand_distances[label].append(None)
        
        for label, distances in hand_distances.items():
            # Remove None values and corresponding timestamps for plotting
            valid_data = [(t, d) for t, d in zip(timestamps[:len(distances)], distances) if d is not None]
            if valid_data:
                valid_timestamps, valid_distances = zip(*valid_data)
                self.ax_distance.plot(valid_timestamps, valid_distances, label=f"{label} distance")
        
        # Mark current frame
        if current_frame < len(timestamps):
            self.ax_distance.axvline(x=timestamps[current_frame], color='red', linestyle='--', alpha=0.7)
        
        self.ax_distance.set_title("Hand Distance Over Time")
        self.ax_distance.set_xlabel("Time (s)")
        self.ax_distance.set_ylabel("Distance (cm)")
        self.ax_distance.legend()
        self.ax_distance.grid(True, alpha=0.3)
    
    def plot_motion_timeline(self, frames, current_frame):
        if not frames:
            return
        
        timestamps = [frame['timestamp'] - frames[0]['timestamp'] for frame in frames]
        
        hand_motions = defaultdict(list)
        for frame in frames:
            if 'hands' in frame:
                for hand in frame['hands']:
                    label = hand.get('hand_label', 'Unknown')
                    if 'motion' in hand and 'overall_motion_magnitude' in hand['motion']:
                        motion = hand['motion']['overall_motion_magnitude']
                        hand_motions[label].append(motion)
                    else:
                        # Pad with 0 for frames without motion data
                        hand_motions[label].append(0)
        
        for label, motions in hand_motions.items():
            if len(motions) > 0:
                self.ax_motion.plot(timestamps[:len(motions)], motions, label=f"{label} motion")
        
        # Mark current frame
        if current_frame < len(timestamps):
            self.ax_motion.axvline(x=timestamps[current_frame], color='red', linestyle='--', alpha=0.7)
        
        self.ax_motion.set_title("Motion Magnitude Over Time")
        self.ax_motion.set_xlabel("Time (s)")
        self.ax_motion.set_ylabel("Motion (cm/s)")
        self.ax_motion.legend()
        self.ax_motion.grid(True, alpha=0.3)
    
    def plot_palm_measurements(self, frame_data):
        all_measurements = {}
        
        if 'hands' in frame_data:
            for hand in frame_data['hands']:
                hand_label = hand.get('hand_label', 'Unknown')
                if 'palm_measurements' in hand:
                    measurements = hand['palm_measurements']
                    for measurement_name, value in measurements.items():
                        key = f"{hand_label}_{measurement_name}"
                        all_measurements[key] = value
        
        if all_measurements:
            names = list(all_measurements.keys())
            values = list(all_measurements.values())
            
            bars = self.ax_measurements.bar(range(len(names)), values)
            self.ax_measurements.set_xticks(range(len(names)))
            self.ax_measurements.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
            self.ax_measurements.set_title("Palm Measurements (Current Frame)")
            self.ax_measurements.set_ylabel("Pixels")
            
            # Color bars by hand
            for i, bar in enumerate(bars):
                if 'Left' in names[i]:
                    bar.set_color('red')
                elif 'Right' in names[i]:
                    bar.set_color('blue')
        else:
            self.ax_measurements.text(0.5, 0.5, 'No measurements available', 
                                    transform=self.ax_measurements.transAxes, 
                                    ha='center', va='center')
    
    def toggle_playback(self):
        if not self.current_file_data or 'sequence' not in self.current_file_data:
            return
        
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")
            if self.animation:
                self.animation.event_source.stop()
        else:
            self.playing = True
            self.play_button.config(text="Pause")
            self.start_animation()
    
    def start_animation(self):
        frames = self.current_file_data['sequence']
        
        def animate(frame_idx):
            if not self.playing or frame_idx >= len(frames):
                self.playing = False
                self.play_button.config(text="Play")
                return
            
            self.current_frame_idx = frame_idx
            self.frame_var.set(frame_idx)
            self.visualize_frame()
        
        # Calculate interval based on speed
        base_interval = 100  # ms
        interval = int(base_interval / self.speed_var.get())
        
        self.animation = FuncAnimation(self.fig, animate, frames=len(frames), 
                                     interval=interval, repeat=False)
        self.canvas.draw()
    
    def reset_animation(self):
        self.playing = False
        self.current_frame_idx = 0
        self.frame_var.set(0)
        self.play_button.config(text="Play")
        if self.animation:
            self.animation.event_source.stop()
        self.visualize_frame()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GestureVisualizer()
    app.run()