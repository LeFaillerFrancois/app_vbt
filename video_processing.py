# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:10:07 2026

@author: francois
"""
import cv2
import deeplabcut
import os
import utils
from PIL import Image, ImageTk
from tkinter import messagebox
import threading
import time

class VideoProcessor:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.cap = None
        self.after_id = None
        self.is_playing = False
        self.output_width = 480
        self.output_height = 640
        self.fps = None
        self.plate_size = None
        
    def add_video_frequency(self, video_frequency):
        self.fps = video_frequency
        # print("fps set to :",self.fps)
    
    def add_real_plate_size(self, real_plate_size):
        self.plate_size = real_plate_size
        # print("plate size set to :",self.plate_size)


    def analyze_video_async(self, video_path, progress_callback, on_finish_callback):
        """Start the analysis in a separate thread to not freeze the gui"""
        def run():
            try:
                # 1. Resize
                progress_callback(0, "Resizing video...")
                resized_path, total_frames = self._resize_video(video_path,progress_callback)
                
                # 2. DeepLabCut Analysis (longest computation)
                progress_callback(40, "DeepLabCut Analysis (Inference)...")
                
                # Fake progress bar
                self.stop_fake_progress = False
                def simulate_dlc_progress():
                    current_percent = 40
                    estimated_it_s = 40
                    estimated_time = round(total_frames / estimated_it_s)
                    count = 0 
                    while not self.stop_fake_progress and current_percent < 79:
                        time.sleep(estimated_time/40)  
                        if count < total_frames:
                            count += estimated_it_s
                        elif count >= total_frames : 
                            count = total_frames
                        increment = (80 - current_percent) / 15
                        current_percent += increment
                        progress_callback(int(current_percent), f"DeepLabCut Analysis estimated percentage for 40it/s ... \n"
                                          f"Analyzing Frame : {count}/{total_frames} frames")
                
                sim_thread = threading.Thread(target=simulate_dlc_progress, daemon=True)
                sim_thread.start()
                deeplabcut.analyze_videos(self.config_path, [resized_path], save_as_csv=False)
                self.stop_fake_progress = True
                
                # 3. Filtering
                progress_callback(80, "Filtering predictions...")
                deeplabcut.filterpredictions(self.config_path, resized_path)
                
                # 4. Create labeled video
                progress_callback(90, "Create labeled video...")
                deeplabcut.create_labeled_video(
                    self.config_path, resized_path, 
                    videotype="mp4", filtered=True
                )
                progress_callback(100, "Finished Analysis !")
                # Find the created video path
                labeled_video = resized_path.replace(".mp4", "DLC_mobnet_100_force_vitesse_powerFeb18shuffle1_48000_filtered_labeled.mp4")
                csv_path = resized_path.replace(".mp4", "DLC_mobnet_100_force_vitesse_powerFeb18shuffle1_48000_filtered.csv")
                result = utils.read_analyse_csv_routine(csv=csv_path, video_frequency=self.fps, real_plate_size=self.plate_size)
                result["video_path"] = labeled_video
                
                # return
                on_finish_callback(result)
                
            except Exception as e:
                print(f"Erreur durant l'analyse : {e}")
                on_finish_callback(None)

        threading.Thread(target=run, daemon=True).start()

    def _resize_video(self, video_path, progress_callback):
        """Resize method, to compute faster"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = os.path.splitext(video_path)[0] + '_resized.mp4'
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.output_width, self.output_height))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            resized_frame = cv2.resize(frame, (self.output_width, self.output_height))
            out.write(resized_frame)
            count += 1
            if count % 10 == 0: # Update every 10 frames 
                percent = int((count / total_frames) * 40) 
                progress_callback(percent, f"Resizing frame : {count}/{total_frames} frames")
        
        cap.release()
        out.release()
        return output_path, total_frames

    def start_playback(self, video_path, label_widget):
        """Start the video player"""
        self.stop_playback()
        self.cap = cv2.VideoCapture(video_path)
        self.is_playing = True
        self._update_frame(label_widget)

    def _update_frame(self, label_widget):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Conversion OpenCV (BGR) to Tkinter (RGB/PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                label_widget.imgtk = imgtk 
                label_widget.configure(image=imgtk)
                
                self.after_id = label_widget.after(30, lambda: self._update_frame(label_widget))
            else:
                self.stop_playback()

    def stop_playback(self, label_widget=None):
        """Stop and release ressources"""
        self.is_playing = False
        if self.after_id:
            if label_widget:
                label_widget.after_cancel(self.after_id)
            self.after_id = None 
        if self.cap:
            self.cap.release()
            self.cap = None