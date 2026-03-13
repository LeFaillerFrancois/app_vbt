# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:08:19 2026

@author: francois
"""

import cv2
from PIL import Image, ImageTk
from dlclive import DLCLive, Processor
import tkinter as tk
from collections import deque
import src.utils as utils

# TODO : finish this page

################# WORK IN PROGRESS ############################################
class WebcamProcessor:
    def __init__(self, model_path="dlc-models/exported-models/DLC_force_vitesse_power_mobilenet_v2_1.0_iteration-0_shuffle-1"):
        self.model_path = model_path
        self.cap = None
        self.dlc_live = None
        self.after_id = None
        self.label_widget = None
        self.is_running = False 

        # Paramètres de tracking
        self.moving_avg_window = 4
        self.confidence_threshold = 0.4
        self.recent_positions = {}
        self.last_good_positions = {}


    def start(self, label_widget):
        self.label_widget = label_widget
        try:
            # On n'initialise DLCLive que si nécessaire
            if self.dlc_live is None:
                self.dlc_live = DLCLive(self.model_path,processor=Processor())
                self.dlc_live.init_inference() 
                print("DLCLive initialized:", self.dlc_live.sess is not None)
                print("Modèle DLC chargé et initialisé.")
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            
            self.is_running = True 
            self.update_frame()
            return True
        except Exception as e:
            print(f"Erreur d'initialisation : {e}")
            return False
        
    def process_point(self, frame, idx, x, y, conf):
        """
        Process the point and update recent positions.
        """

        if idx not in self.recent_positions:
            self.recent_positions[idx] = deque(maxlen=self.moving_avg_window)

        if conf >= self.confidence_threshold:
            # Add the new position
            self.recent_positions[idx].append((x, y))
            self.last_good_positions[idx] = (x, y)
        elif idx in self.last_good_positions:
            # Low confidence → use the last reliable position
            self.recent_positions[idx].append(self.last_good_positions[idx])
        else:
            # No reliable data → skip this point
            return

        # Average of recent positions
        avg_pos = utils.get_average_position(self.recent_positions[idx])
        # Display the point
        if avg_pos:
            utils.draw_tracker_point(frame, avg_pos[0], avg_pos[1])

        # Update the smoothed buffer (for future analysis)
        # if idx not in recent_smooth_positions:
        #     recent_smooth_positions[idx] = deque(maxlen=50)
        # recent_smooth_positions[idx].append((avg_x, avg_y))


    def update_frame(self):
        if not self.is_running or not self.label_widget.winfo_exists():
            return
    
        ret, frame = self.cap.read()
        if ret:
            try:
                # Sécurité si le modèle n'est pas prêt
                if self.dlc_live is not None:
                    pose = self.dlc_live.get_pose(frame)
                    if pose is not None:
                        for idx, point in enumerate(pose):
                            x, y, conf = point
                            self.process_point(frame, idx, x, y, conf)
            except Exception as e:
                print(f"Erreur d'inférence : {e}")
                # On continue quand même pour afficher l'image brute si l'IA bug    
                
            # Conversion pour l'affichage
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.label_widget.imgtk = imgtk # Référence pour éviter le Garbage Collector
            self.label_widget.configure(image=imgtk)
            
            self.after_id = self.label_widget.after(10, self.update_frame)
    

    def stop(self):
        self.is_running = False 
        if self.after_id:
            self.label_widget.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.label_widget:
            self.label_widget.config(image='')
            
