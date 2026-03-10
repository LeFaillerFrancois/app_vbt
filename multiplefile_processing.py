# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 00:48:40 2026

@author: francois
"""
import utils
import numpy as np

class MultipleFileProcessor:
    def __init__(self):
        self.files_data = [] # List of dicts: {"path": str, "load": float}
        self.loads = []
        self.results = []
        self.result = None
        self.fps = None
        self.plate_size = None
    
    def add_video_frequency(self, video_frequency):
        self.fps = video_frequency
    
    def add_real_plate_size(self, real_plate_size):
        self.plate_size = real_plate_size


    def add_file(self, file_path, load):
        self.files_data.append({"path": file_path, "load": load})

    def clear_data(self):
        self.files_data = []
    
    def analyse_file(self,path):
        self.result = utils.read_analyse_csv_routine(csv=path, video_frequency =self.fps ,real_plate_size =self.plate_size)
        return self.result

    def compute_profile(self):
        #max_velocities = []
        for item in self.files_data:
            #self.results.append(utils.read_analyse_csv_routine(csv=item["path"]))
            self.results.append(self.analyse_file(path=item["path"]))
            self.loads.append(float(item["load"]))
        self.loads=np.array(self.loads)
        mean_speeds, a, b, predict, F0_1st, V0_1st, rcarre, min_max_bench, max_max_bench, squat_dead_max = utils.compute_load_velocity_profil(self.results, self.loads)
        # 5. Regression
        #Yprint(f"Vitesse moy de la première rep : {self.results[0]['mean_speed'][0]}")
        return self.results, mean_speeds, a, b, predict, F0_1st, V0_1st, rcarre, min_max_bench, max_max_bench, squat_dead_max