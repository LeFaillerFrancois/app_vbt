# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:06:28 2026

@author: francois
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from src.video_processing import VideoProcessor
from src.webcam_processing import WebcamProcessor
from src.multiplefile_processing import MultipleFileProcessor
import src.utils as utils
import numpy as np
import pandas as pd
import json
import os

# Window to manually edit automatic rep detection in case it didnt work as intended
# Usable even for a single file if provided as list
class RepEditorWindow(tk.Toplevel):
    def __init__(self, parent, results_list, callback, fps = 30, plate_size = 0.45):
        super().__init__(parent)
        self.title("Repetition manual validation")
        self.geometry("1100x800")
        self.results_list = results_list
        self.fps = fps
        self.plate_size = plate_size
        self.callback = callback
        self.current_idx = 0
        
        self.scatters = {}
        self.currently_dragging = None
        self.next_type_idx = 0 
        self.type_sequence = []
        
        self.setup_ui()
        self.load_file(0)

    def setup_ui(self):
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.lbl_help = tk.Label(self.top_frame, text="", fg="#555", font=('Helvetica', 9, 'italic'))
        self.lbl_help.pack()
        
        self.top_lbl = tk.Label(self.top_frame, text="", font=("Helvetica", 12, "bold"))
        self.top_lbl.pack(pady=5)
        
        self.lbl_next = tk.Label(self.top_frame, text="", font=('Helvetica', 10, 'bold'))
        self.lbl_next.pack()

        # Plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        
        # Navigation
        self.nav_frame = tk.Frame(self)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        tk.Button(self.nav_frame, text="◄ Previous", command=self.prev, width=15).pack(side=tk.LEFT, padx=50)
        tk.Button(self.nav_frame, text="FINISH and COMPUTE", bg="#4CAF50", fg="white", font=('Helvetica', 10, 'bold'),
                  command=self.finish).pack(side="left", expand=True)
        tk.Button(self.nav_frame, text="Next ►", command=self.next, width=15).pack(side=tk.RIGHT, padx=50)

        # event connection
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def load_file(self, idx):
        self.current_idx = idx
        res = self.results_list[idx]
        
        mode = res["mode"]
        if mode == "squat_bench_like":
            self.type_sequence = ["start", "peaks", "start_concentric", "end"]
            help_txt = "Double-clic: Add (Cycle Start->Peak->Start Concentric->End) | Shift+Clic: Delete | Hold: Move"
        else:
            self.type_sequence = ["start", "peaks", "end"]
            help_txt = "Double-clic: Add (Cycle Start->Peak->End) | Shift+Clic: Delete | Hold: Move"
        
        self.lbl_help.config(text=help_txt)
        self.top_lbl.config(text=f"File {idx+1}/{len(self.results_list)} : {res.get('filename', 'file')}")
        
        # Reset plots
        self.ax1.clear()
        self.ax2.clear()
        self.next_type_idx = 0
        self.update_status_label()
        
        self.ax1.plot(res["y_filt"], color='gray', alpha=0.4, label="Position y (m)")
        self.ax2.plot(res["y_speed"], color='gray', alpha=0.4, label="Speed y (m/s)")
        
        self.ax1.set_xlabel("Time [Frame]")
        self.ax2.set_xlabel("Time [Frame]")
        self.ax1.set_ylabel("y [m]")
        self.ax2.set_ylabel("y [m/s]")
        self.ax1.set_title("Bar Height")
        self.ax2.set_title("Bar Vertical Speed")
        
        # Load the first file and plot datas
        self.update_plot_data()

    def update_status_label(self):
        nxt_type = self.type_sequence[self.next_type_idx]
        colors = {"start": "green", "peaks": "red", "start_concentric": "pink", "end": "blue"}
        self.lbl_next.config(text=f"Next point to add : {nxt_type.upper()}", fg=colors.get(nxt_type, "black"))
    
    def update_plot_data(self):
        data = self.results_list[self.current_idx]
        colors = {"start": "green", "peaks": "red", "start_concentric": "pink", "end": "blue"}
        
        # Reset existing points
        for s1, s2 in self.scatters.values():
            s1.remove()
            s2.remove()
        
        self.scatters = {}
        for key in self.type_sequence:
            if key not in data: continue
            
            idxs = data[key]
            y_pos = [data["y_filt"][i] for i in idxs]
            y_spd = [data["y_speed"][i] for i in idxs]
            
            # Creating new event points to use
            s1 = self.ax1.scatter(idxs, y_pos, color=colors.get(key, "black"), s=70, picker=8, zorder=5)
            s2 = self.ax2.scatter(idxs, y_spd, color=colors.get(key, "black"), label=key, s=70, picker=8, zorder=5)
            self.scatters[key] = (s1, s2)

        self.ax2.legend(loc='upper right', framealpha=0.4, draggable=True)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes is None: return
        data = self.results_list[self.current_idx]
        #print(f"DEBUG: Next type to add: {self.type_sequence[self.next_type_idx]} | Keys in data: {data.keys()}")
        # --- ADDING a point (Double clic) ---
        if event.dblclick:
            new_x = int(round(event.xdata))
            new_x = max(0, min(len(data["y_filt"])-1, new_x))
            
            current_type = self.type_sequence[self.next_type_idx]
            data[current_type].append(new_x)
            data[current_type].sort()
            
            # Cycle through every type of point according to exercice
            self.next_type_idx = (self.next_type_idx + 1) % len(self.type_sequence)
            
            self.update_status_label()
            self.update_plot_data()
            return

        # --- DETECTION / SUPPRESSION ---
        for name, (s1, s2) in self.scatters.items():
            cont, ind = s1.contains(event) if event.inaxes == self.ax1 else s2.contains(event)
            if cont:
                pt_idx = ind['ind'][0]
                
                if event.key == 'shift':
                    data[name].pop(pt_idx)
                    self.update_plot_data()
                    return
                
                self.currently_dragging = (name, pt_idx)
                break

    def on_motion(self, event):
        if self.currently_dragging is None or event.inaxes is None: return
        name, pt_idx = self.currently_dragging
        res = self.results_list[self.current_idx]
        
        new_x = int(round(event.xdata))
        new_x = max(0, min(len(res["y_filt"])-1, new_x))
        
        # Update Data
        res[name][pt_idx] = new_x
        
        # Update Visuals
        s1, s2 = self.scatters[name]
        off1, off2 = s1.get_offsets(), s2.get_offsets()
        off1[pt_idx] = [new_x, res["y_filt"][new_x]]
        off2[pt_idx] = [new_x, res["y_speed"][new_x]]
        s1.set_offsets(off1)
        s2.set_offsets(off2)
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.currently_dragging:
            name, _ = self.currently_dragging
            self.results_list[self.current_idx][name].sort()
        self.currently_dragging = None

    def next(self):
        if self.current_idx < len(self.results_list)-1:
            self.load_file(self.current_idx + 1)

    def prev(self):
        if self.current_idx > 0:
            self.load_file(self.current_idx - 1)
            
    # When closing (finish and compute button) update de result dictionnary with 
    # The new event points
    def finish(self):
        for res in self.results_list:
            
            if res["mode"] == "squat_bench_like":
                res = utils.update_result(res, res["start"], res["peaks"], res["end"], res["start_concentric"], fps = self.fps, real_plate_size = self.plate_size)
            else: 
                res = utils.update_result(res, res["start"], res["peaks"], res["end"], fps = self.fps, real_plate_size = self.plate_size)
            
        self.callback(self.results_list)
        self.destroy()

# Actual app
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepLabCut Bar Tracker")
        self.state('zoomed')
        #self.geometry("800x600")

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        # Implemented pages list
        for F in (StartPage, VideoProcessingPage, WebcamProcessingPage, MultipleFilePage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        # Default page, menu
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """show a page and stop the others"""
        frame = self.frames[page_name]
        
        for f in self.frames.values():
            if hasattr(f, 'stop_process'):
                f.stop_process()

        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        # background color
        super().__init__(parent, bg="#f0f2f5") 
        self.controller = controller

        # --- Full screen container ---
        self.main_bg = tk.Frame(self, bg="#f0f2f5")
        self.main_bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        # --- Mid panel (Menu) ---
        menu_frame = tk.Frame(self.main_bg, bg="#f0f2f5")
        menu_frame.place(relx=0.55, rely=0.4, anchor="center")

        # --- TITlE & DESIGN ---
        # tk.Label(menu_frame, text="🏋️BARTRACKER", 
        #          font=("Helvetica", 32, "bold"), 
        #          fg="#1a2a3a", bg="#f0f2f5").pack(pady=(0, 5))
        # Handmade logo lmao ;)
        self.logo_img = tk.PhotoImage(file="read_me_preview/Logo.png")
        logo_label = tk.Label(menu_frame, image=self.logo_img, bg="#f7f9fc")
        logo_label.pack()
        
        tk.Label(menu_frame, text="Velocity Based Training & Analysis System", 
                 font=("Helvetica", 12), 
                 fg="#5a6a7a", bg="#f0f2f5").pack(pady=(0, 40))

        # --- BUTTON STYLE ---
        buttons_config = [
            ("🎥  VIDEO ANALYSIS", "VideoProcessingPage", "#3498db"),
            ("📊  F-V PROFILING", "MultipleFilePage", "#2ecc71"),
            ("📷  LIVE WEBCAM (WORK IN PROGRESS)", "WebcamProcessingPage", "#95a5a6")
        ]

        for text, route, color in buttons_config:
            btn = tk.Button(
                menu_frame, 
                text=text,
                command=lambda r=route: controller.show_frame(r),
                font=("Helvetica", 11, "bold"),
                bg=color, 
                fg="white",
                activebackground="#2c3e50", 
                activeforeground="white",
                relief="flat",
                width=40,
                height=2,
                cursor="hand2"
            )
            btn.pack(pady=12)
            
            # Hovering color changing
            btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg="#2c3e50")) 
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

        # --- FOOTER ---
        footer_label = tk.Label(self.main_bg, text="2026 BarTracker | Le Failler François", 
                                font=("Helvetica", 9), fg="#95a5a6", bg="#f0f2f5")
        footer_label.pack(side="bottom", pady=20)

class VideoProcessingPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        # Video processor logic in another .py
        self.processor = VideoProcessor() # Default one
        self.results_list = [0] # To be able to be access with repeditorwindow
        self.plate_size_var = tk.StringVar(value="0.45") # (m) Plate size, default value 0.45m (if calibrated)
        self.fps_var = tk.StringVar(value="30")         # (fps) Video frame rate, default value 30fps
        self.model_check = tk.BooleanVar(value = False)
        self.size_name_variable = tk.StringVar(value="Plate Size (m):")
        self.setup_layout()

    def setup_layout(self):
        # --- LEFT PANEL (Video player & Buttons) ---
        self.left_panel = tk.Frame(self, width=480, bg="#f0f0f0")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=(5,25))

        tk.Label(self.left_panel, text="VIDEO PLAYER", font=("Helvetica", 10, "bold")).pack(pady=2)
        
        # Video player
        self.video_container = tk.Frame(self.left_panel, width=480, height=640, bg="black")
        self.video_container.pack(pady=2)
        self.video_container.pack_propagate(False)
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Buttons
        self.btn_analyze = tk.Button(self.left_panel, text="Load video", 
                                     command=self.load_video, bg="#2196F3", fg="white", height=1)
        self.btn_analyze.pack(fill=tk.X, pady=2)

        self.btn_edit = tk.Button(self.left_panel, text="Manual Rep Detection", 
                                  command=self.open_rep_editor, state="disabled")
        self.btn_edit.pack(fill=tk.X, pady=2)
        self.bouton_check_calisthenics_model = tk.Checkbutton(self.left_panel, text="Experimental : Calisthenics model, check to enable",
                                           variable=self.model_check, command=self.change_model_label_text)
        self.bouton_check_calisthenics_model.pack(fill=tk.X, pady=2)
        
        # Progression bar
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_bar = ttk.Progressbar(self.left_panel, variable=self.progress_var, maximum=100)
        self.status_label = tk.Label(self.left_panel, textvariable=self.status_var, font=("Helvetica", 9, "italic"))
        
        tk.Button(self.left_panel, text="Main Menu", 
                  command=lambda: self.controller.show_frame("StartPage")).pack(fill=tk.X, pady=5)#side=tk.BOTTOM, fill=tk.X)
        
        # Config frame (fps and plate size)
        config_frame = ttk.LabelFrame(self.left_panel, text="Settings", padding=5)
        config_frame.pack(fill="x", pady=5)
        tk.Label(config_frame, textvariable=self.size_name_variable).grid(row=0, column=0, sticky="w", padx=2)
        self.ent_plate = tk.Entry(config_frame, textvariable=self.plate_size_var, width=8)
        self.ent_plate.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(config_frame, text="Video FPS:").grid(row=1, column=0, sticky="w", padx=2)
        self.ent_fps = tk.Entry(config_frame, textvariable=self.fps_var, width=8)
        self.ent_fps.grid(row=1, column=1, padx=5, pady=2)
        
        # --- RIGHT PANEL (Plots) ---
        self.right_panel = tk.Frame(self, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=(5,20))

        tk.Label(self.right_panel, text="PERFORMANCE ANALYSIS", font=("Helvetica", 10, "bold")).pack(pady=5)

        # 3 plots : rom, mean speed, bar path
        self.fig, (self.ax_rom, self.ax_speed, self.ax_path) = plt.subplots(3, 1, figsize=(5, 7), constrained_layout=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="y", expand=False)
        
        # Saving button : Figures
        self.btn_save_fig = tk.Button(self.right_panel, text="💾 Save Figures", 
                                      command=self.save_figures, state="disabled", bg="#FF9800", fg="white")
        self.btn_save_fig.pack(fill=tk.X, pady=2)

        # Saving button : Data
        self.btn_save_data = tk.Button(self.right_panel, text="📊 Save Data (CSV/Json)", 
                                       command=self.save_data, state="disabled", bg="#4CAF50", fg="white")
        self.btn_save_data.pack(fill=tk.X, pady=2)
        
        
    def change_model_label_text(self):
        new_text = "Height (m):" if self.model_check.get() else "Plate Size (m):"
        self.size_name_variable.set(new_text)
        
    
    def load_processor(self):
        if self.model_check.get() :
            self.processor = VideoProcessor(config_path="calisthenics_model/config.yaml") 
            print("calisthenics model")
        else : 
            self.processor = VideoProcessor()
            print("powerlifting model")

    def load_video(self):
        self.load_processor()
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if video_path:
            current_plate_size = float(self.plate_size_var.get())
            current_fps = float(self.fps_var.get())
            self.processor.add_real_plate_size(current_plate_size)
            self.processor.add_video_frequency(current_fps)
            
            self.progress_bar.pack(fill=tk.X, pady=5)
            self.status_label.pack()
            self.btn_analyze.config(state="disabled")
            # Video analysis (async in order to have a working gui while its analysing)
            self.processor.analyze_video_async(
                video_path, 
                self._update_ui_progress,  # Update progress bar
                self._on_finished,
            )
    
    # Update progress bar
    def _update_ui_progress(self, value, status_text):
        self.after(0, lambda: self.progress_var.set(value))
        self.after(0, lambda: self.status_var.set(status_text))

    def _on_finished(self, result):
        self.after(0, self.progress_bar.pack_forget)
        self.after(0, lambda: self.btn_analyze.config(state="normal"))
        self.after(0, lambda: self.btn_edit.config(state="normal"))
        
        if result:
            messagebox.showinfo("Alert", "Done !")
            self.results_list[0]=result 
            # Start the video player
            self.processor.start_playback(result['video_path'], self.video_label)
            self.update_plots(result)

            self.after(0, lambda: self.btn_save_fig.config(state="normal"))
            self.after(0, lambda: self.btn_save_data.config(state="normal"))
            #print("end : ", result["end"])
            #print("rep y : ", result["reps_y"])


    def update_plots(self, data):
        """Updating plots"""
        # Reset plots
        self.ax_rom.clear()
        self.ax_speed.clear()
        self.ax_path.clear()
        
        # Compute rom over reps
        reps = np.arange(len(data["ROM"]))+1
        tick_label = reps
        
        # Range of motion plot
        self.ax_rom.bar(reps, data["ROM"], tick_label=tick_label, color='skyblue')
        self.ax_rom.set_ylabel("ROM (m)")
        self.ax_rom.set_title("Range of Motion over reps")

        # Bar speed plot
        self.ax_speed.bar(reps, data['mean_speed'], tick_label=tick_label, color='red')
        self.ax_speed.set_ylabel("Speed (m/s)")
        self.ax_speed.set_title("Mean speed over reps")

        # Bar path plot
        normalized_reps_y = data["reps_y"]
        normalized_reps_x = data["reps_x"]
        mean_traj_y = data["mean_traj_y"]
        mean_traj_x = data["mean_traj_x"]
        mean_traj_x=mean_traj_x-mean_traj_x[0]
        mean_traj_y=mean_traj_y-mean_traj_y[0]
        plt.plot(mean_traj_x, mean_traj_y,linestyle='--',linewidth=2,color="red")
        for i in range(len(reps)):
            self.ax_path.plot(normalized_reps_x[i,:]-normalized_reps_x[i,0], normalized_reps_y[i,:]-normalized_reps_y[i,0], alpha=0.3)        
        self.ax_path.plot(mean_traj_x, mean_traj_y,linestyle='--',linewidth=2,color="red")
        self.ax_path.set_title("Bar path")
        self.ax_path.set_xlabel("Width (m)")
        self.ax_path.set_ylabel("Height (m)")
        self.ax_path.set_xlim(-0.5, 0.5)
        self.ax_path.relim()
        self.ax_path.autoscale_view(scalex=False, scaley=True)
        self.ax_path.set_aspect('equal', adjustable='box')
        if data["mode"] == "squat_bench_like":
            if not self.ax_path.yaxis_inverted():
                self.ax_path.invert_yaxis()

        self.canvas.draw()

    def open_rep_editor(self):
        """Open the rep editor to manually detect if needed"""
        if not self.results_list:
            return
        editor = RepEditorWindow(self, self.results_list, self.on_editor_closed, float(self.fps_var.get()), float(self.plate_size_var.get()))
        editor.grab_set() 

    def on_editor_closed(self, updated_results):
        """Callback when rep editors is closed"""
        self.results_list = updated_results
        messagebox.showinfo("Rep editor", "Synchronised data !")
        # Update plots according to the new manual detection
        self.update_plots(updated_results[0])
        #print("updated ends : ", updated_results[0]["end"])
        #print("updated rep y : ", updated_results[0]["reps_y"])
        
    def save_figures(self):
        if not self.results_list[0]: return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG info", "*.png"), ("PDF info", "*.pdf"), ("SVG info", "*.svg")],
            title="Save Analysis Figures"
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figures saved to:\n{file_path}")    

    def ask_export_format(self):
        """Open a window to select export format."""
        self.chosen_format = None
        win = tk.Toplevel(self)
        win.title("Export Format")
        win.geometry("300x150")
        win.grab_set() 

        tk.Label(win, text="Choose your export format:", font=("Helvetica", 10, "bold")).pack(pady=10)

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=5)

        def select(fmt):
            self.chosen_format = fmt
            win.destroy()

        tk.Button(btn_frame, text="CSV (Padding, incomplete data)", width=18, command=lambda: select("csv")).pack(pady=2)
        tk.Button(btn_frame, text="JSON (Full Dict)", width=18, command=lambda: select("json")).pack(pady=2)
        
        self.wait_window(win)  
        return self.chosen_format
    
    def save_data(self):
        data = self.results_list[0]
        if not data: return
        
        fmt = self.ask_export_format()
        if not fmt: return 

        if fmt == "csv":
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV files", "*.csv")])
            if file_path:
                self._export_to_csv_padding(data, file_path)
        
        elif fmt == "json":
            file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                   filetypes=[("JSON files", "*.json")])
            if file_path:
                self._export_to_json(data, file_path)

    def _export_to_csv_padding(self, data, file_path):
        try:
            # 1. Reps metrics
            df_reps = pd.DataFrame({
                "Repetition": np.arange(len(data["ROM"])) + 1,
                "ROM_m": data["ROM"],
                "Mean_Speed_ms": data["mean_speed"],
                "Max_Speed_ms": data["max_speed"]
            })
            # 2. Reps Trajectories 
            df_signals = pd.DataFrame({
                "Reps_x": data["x_filt"],
                "Reps_y": data["y_filt"],
            })
            # Padding
            df_final = pd.concat([df_reps, df_signals], axis=1)
            df_final.to_csv(file_path, index=False, sep=";")
            messagebox.showinfo("Success", "CSV exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"CSV Export failed: {e}")

    def _export_to_json(self, data, file_path):
        # On définit un encodeur personnalisé
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # On utilise l'argument 'cls' pour passer notre encodeur
                json.dump(data, f, cls=NpEncoder, indent=4)
            
            messagebox.showinfo("Success", "JSON exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"JSON Export failed: {e}")

    def stop_process(self):
        self.processor.stop_playback()


# TODO : Work in progress page
# As it is now, only load a tfl model via deeplabcut live to start tracking via webcam 
# No analysis behing yet
class WebcamProcessingPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processor = WebcamProcessor()
        self.is_running = False

        self.webcam_display = tk.Label(self)
        self.webcam_display.pack()

        self.btn_toggle = tk.Button(self, text="Start Webcam", command=self.toggle_webcam)
        self.btn_toggle.pack(pady=10)
        
        tk.Button(self, text="Main Menu", command=lambda: controller.show_frame("StartPage")).pack()

    def toggle_webcam(self):
        if not self.is_running:
            if self.processor.start(self.webcam_display):
                self.is_running = True
                self.btn_toggle.config(text="Stop Webcam")
        else:
            self.stop_process()

    def stop_process(self):
        self.processor.stop()
        self.is_running = False
        self.btn_toggle.config(text="Start Webcam")
        self.webcam_display.config(image='')
        

# Load multiple csv to compute F-V profile
class MultipleFilePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        # Logic in another .py Maybe more logic could be passed to the processor, this code is quite heavy
        self.processor = MultipleFileProcessor()
        
        self.plate_size_var = tk.StringVar(value="0.45") # (m) Plate size, default value 0.45m (if calibrated)
        self.fps_var = tk.StringVar(value="30")         # (fps) Video frame rate, default value 30fps
        self.result_to_export = {}
        # UI Layout: Left side for controls/list, Right side for Plot
        # TODO : setup layout ?
        # --- LEFT PANEL (file list view & Buttons) ---
        self.left_panel = ttk.Frame(self, width=850) 
        self.left_panel.pack_propagate(False) 
        self.left_panel.pack(side="left",fill="y", padx=(100,100), pady=(10,400))
        tk.Label(self.left_panel, text="F-V Profiling", font=("Helvetica", 16, "bold")).pack(pady=5)
        
        # File List View
        self.tree = ttk.Treeview(self.left_panel, columns=("File", "Load"), show='headings', height=6)
        self.tree.heading("File", text="File")
        self.tree.heading("Load", text="Load (kg)")
        self.tree.column("Load", width=70, anchor="center")
        self.tree.pack(fill="x", pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.left_panel)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Add CSV", command=self.add_trial).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Clear", command=self.clear_list).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Compute", command=self.run_analysis).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Main Menu", command=lambda: controller.show_frame("StartPage")).pack(side="left", padx=2)

        # Results Label
        self.result_label = tk.Label(self.left_panel, text="Results will appear here", font=("Courier", 9), 
                                     justify="left", bg="white", relief="sunken", anchor="nw", padx=5, pady=5)
        self.result_label.pack(fill="both", expand=True, pady=10)
        
        # Config (plate size and fps)
        config_frame = ttk.LabelFrame(self.left_panel, text="Settings", padding=5)
        config_frame.pack(fill="x", pady=5)
        tk.Label(config_frame, text="Plate Size (m):").grid(row=0, column=0, sticky="w", padx=2)
        self.ent_plate = tk.Entry(config_frame, textvariable=self.plate_size_var, width=8)
        self.ent_plate.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(config_frame, text="Video FPS:").grid(row=1, column=0, sticky="w", padx=2)
        self.ent_fps = tk.Entry(config_frame, textvariable=self.fps_var, width=8)
        self.ent_fps.grid(row=1, column=1, padx=5, pady=2)
        
        
        # --- RIGHT PANEL (plots) ---
        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=(5,400))

        # Matplotlib Figure Setup
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Saving button : Figures
        self.btn_save_fig = tk.Button(self.right_panel, text="💾 Save Figures", 
                                      command=self.save_figures, state="disabled", bg="#FF9800", fg="white")
        self.btn_save_fig.pack(fill=tk.X, pady=2)

        # Saving button : Data
        self.btn_save_data = tk.Button(self.right_panel, text="📊 Save Data (CSV/Json)", 
                                       command=self.save_data, state="disabled", bg="#4CAF50", fg="white")
        self.btn_save_data.pack(fill=tk.X, pady=2)

    def add_trial(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path: return

        load_win = tk.Toplevel(self)
        load_win.title("Load")
        load_win.geometry("250x120")
        load_win.grab_set()
        
        tk.Label(load_win, text=f"Load (kg) for {os.path.basename(file_path)}:").pack(pady=10)
        entry = tk.Entry(load_win)
        entry.pack()
        entry.focus_set()

        def save():
            try:
                val = float(entry.get())
                self.processor.add_file(file_path, val)
                self.tree.insert("", "end", values=(os.path.basename(file_path), val))
                load_win.destroy()
            except ValueError: messagebox.showerror("Error", "Enter a number")

        tk.Button(load_win, text="OK", command=save).pack(pady=10)
        entry.bind("<Return>", lambda e: save())

    def run_analysis(self):
        if len(self.processor.files_data) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points to compute F-V profile.")
            return
        
        try:
            # TODO : ability to change plate size and fps according to each unique video
            current_plate_size = float(self.plate_size_var.get())
            current_fps = float(self.fps_var.get())
            self.processor.add_real_plate_size(current_plate_size)
            self.processor.add_video_frequency(current_fps)

            all_results = []
            for item in self.processor.files_data:    
                res = self.processor.analyse_file(item["path"]) 
                res["load"] = item["load"]
                res["filename"] = os.path.basename(item["path"])
                all_results.append(res)
                
                
            # Once all csv are analysed, open rep editor to manually validate in case
            RepEditorWindow(self, all_results, self.finalize_profile_calculation)

        except Exception as e:
            messagebox.showerror("Analyse Error", f"Erreur : {str(e)}")

    def finalize_profile_calculation(self, corrected_results):
        """Called after the user clicked on Finish in the rep editor"""
        try:
            self.after(0, lambda: self.btn_save_fig.config(state="normal"))
            self.after(0, lambda: self.btn_save_data.config(state="normal"))
            
            loads = np.array([res["load"] for res in corrected_results])
            range_of_motion = np.mean(utils.compute_ROM_across_csv(corrected_results))
            # Compute the F V profile with the new manual detection 
            load_velocity_profil = utils.compute_load_velocity_profil(corrected_results,loads)
            
            mean_speeds = load_velocity_profil["mean_speeds"]
            regr = load_velocity_profil["regr"]
            # coeff
            rsquared = regr.score(mean_speeds, loads) # R²
            a=regr.coef_ 
            b = regr.intercept_ 
            predict=regr.predict(mean_speeds)

            self.result_to_export["loads"]=loads.tolist()
            self.result_to_export["speeds"]=mean_speeds.reshape(-1).tolist()
            self.result_to_export["y"]=f"{round(a[0,0],1)}x + {round(b[0],1)}"
            self.result_to_export["R"]=f"{rsquared:.3f}"

            # print(range_of_motion)
            
            # 3. Update Text UI (discriminate between squat, bench and deadlift based on ROM
            # and when the concentric is in the rep. Do not work as intended with other exercices
            # but max sbd speeds are exercice dependant anyway
            # Bench
            if (range_of_motion < 0.40) and (corrected_results[-1]["mode"]=="squat_bench_like"):

                output = (f"Maximal load for an almost motionless bench : \n" 
                          f"(0.15m/s) < x < (0.12m/s) :  {round(regr.predict([[0.15]])[0,0],2)} < x < {round(regr.predict([[0.12]])[0,0],2)} kg.\n"
                          f"\n"
                          f"Maximal THEORICAL load if motionless (0m/s) : {b[0]}kg.\n"
                          f"Maximal speed with no load at all : {round(-b[0]/a[0,0],2)}m/s.")
            # Squat
            elif (range_of_motion > 0.40) and (corrected_results[-1]["mode"]=="squat_bench_like"): 
                output = (f"Maximal load for an almost motionless squat : \n"
                          f"(0.31m/s) < x < (0.25m/s) : {round(regr.predict([[0.31]])[0,0],2)} < x < {round(regr.predict([[0.25]])[0,0],2)} kg.\n"
                          f"\n"
                          f"Maximal THEORICAL load if motionless (0m/s) : {b[0]}kg.\n"
                          f"Maximal speed with no load at all : {round(-b[0]/a[0,0],2)}m/s.")
            # Deadlift
            else : 
                output = (f"Maximal load for an almost motionless deadlift : \n"
                          f"(0.32m/s) < x < (0.25m/s) : {round(regr.predict([[0.32]])[0,0],2)} < x < {round(regr.predict([[0.25]])[0,0],2)} kg.\n"
                          f"\n"
                          f"Maximal THEORICAL load if motionless (0m/s) : {b[0]}kg.\n"
                          f"Maximal speed with no load at all : {round(-b[0]/a[0,0],2)}m/s.")

            self.result_label.config(text=output)
            
            # 4. Update Plot
            self.ax.clear()
            self.ax.scatter(mean_speeds, loads, color='blue', label="Mean speed")
            self.ax.plot(mean_speeds, predict, color='red')#, label=f"R²={rsquared:.3f}")
            self.ax.set_xlabel("Speed [m/s]")
            self.ax.set_ylabel("Load [kg]")
            self.ax.text(min(mean_speeds)+0.0*min(mean_speeds),min(loads)+0.010*min(loads),f"R² = {rsquared:.3f}", fontsize=12)
            self.ax.text(min(mean_speeds)+0.0*min(mean_speeds),min(loads)+0.0*min(loads),f"y = {round(a[0,0],1)}x + {round(b[0],1)}", fontsize=12)
            self.ax.set_title("Load Velocity Profile")
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Calcul Error", f"Error during final computation : {str(e)}")
            
    def save_figures(self):
        if not self.result_to_export: return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG info", "*.png"), ("PDF info", "*.pdf"), ("SVG info", "*.svg")],
            title="Save Analysis Figures"
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figures saved to:\n{file_path}")    
            
    def save_data(self):
        data = self.result_to_export
        if not data: return
        
        fmt = self.ask_export_format()
        if not fmt: return 

        if fmt == "csv":
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV files", "*.csv")])
            if file_path:
                self._export_to_csv_padding(data, file_path)
        
        elif fmt == "json":
            file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                   filetypes=[("JSON files", "*.json")])
            if file_path:
                self._export_to_json(data, file_path)
                
    def ask_export_format(self):
        """Open a window to select export format."""
        self.chosen_format = None
        win = tk.Toplevel(self)
        win.title("Export Format")
        win.geometry("300x150")
        win.grab_set() 

        tk.Label(win, text="Choose your export format:", font=("Helvetica", 10, "bold")).pack(pady=10)

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=5)

        def select(fmt):
            self.chosen_format = fmt
            win.destroy()

        tk.Button(btn_frame, text="CSV (Padding, incomplete data)", width=18, command=lambda: select("csv")).pack(pady=2)
        tk.Button(btn_frame, text="JSON (Full Dict)", width=18, command=lambda: select("json")).pack(pady=2)
        
        self.wait_window(win)  
        return self.chosen_format
    
    def _export_to_csv_padding(self, data, file_path):
        try:
            # 1. Reps metrics
            df = pd.DataFrame.from_dict(data)
            df.to_csv(file_path, index=False, sep=";")
            messagebox.showinfo("Success", "CSV exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"CSV Export failed: {e}")

    def _export_to_json(self, data, file_path):
        try:
            # convert numpy arrays into lists 
            clean_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    clean_data[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    clean_data[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                else:
                    clean_data[k] = v
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=4)
            messagebox.showinfo("Success", "JSON exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"JSON Export failed: {e}")
    

    def clear_list(self):
        self.processor.clear_data()
        for i in self.tree.get_children(): self.tree.delete(i)
        self.ax.clear()
        self.canvas.draw()
        self.result_label.config(text="")

    def stop_process(self):
        pass

        
if __name__ == "__main__":
    app = Application()
    app.mainloop()