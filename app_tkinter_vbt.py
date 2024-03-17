# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:14:13 2024

@author: francois
"""
import tkinter as tk
from tkinter import filedialog
import cv2
import deeplabcut
import pandas as pd
#import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import statistics
from scipy.signal import find_peaks
#from sklearn import linear_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Definition of data processing functions

def load_df(csv_file_path):
    dataframe = pd.read_csv(csv_file_path, header=1)
    dataframe = dataframe.drop(0)
    dataframe = dataframe.drop(dataframe.columns[0], axis=1)
    dataframe = dataframe.astype(float)
    dataframe = dataframe.iloc[5:-5]
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def calculate_plate_size(dataframe):
    x2, x1 = dataframe["bas"], dataframe["haut"]
    y2, y1 = dataframe["bas.1"], dataframe["haut.1"]
    plate_size = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    median_plate_size = statistics.median(plate_size)
    return median_plate_size

def calculate_ratio(median_plate_size):
    ratio = real_plate_size / median_plate_size
    return ratio

def find_height_peak(serie):
    peaks, _ = find_peaks(serie, distance=8,prominence=25)
    return peaks

def find_start_end_peak(serie, peaks):
    mean_first_sec = np.mean(serie[:video_frequency])
    mean_first_sec_plus_5_percent = mean_first_sec * 1.05
    threshold = mean_first_sec_plus_5_percent
    list_start_peak = []
    list_end_peak = []
    
    for j in range(len(peaks)):    
        i = peaks[j]
        k = peaks[j]
        while serie.iloc[i] > threshold:
            i -= 1
        while serie.iloc[k] > threshold:
            k += 1
            
        list_start_peak.append(i-10)
        list_end_peak.append(k+8)
    return list_start_peak, list_end_peak

def calculate_concentric_speed(dataframe, peaks, list_start_peak, list_end_peak, ratio):
    if dataframe["milieu.1"].iloc[peaks[0]] < dataframe["milieu.1"].iloc[peaks[0]-8]:
        list_peak = list_start_peak
    else:
        list_peak = list_end_peak
        
    vertical_movement_speed = []
    for peak_index, peak in enumerate(peaks):
        vertical_displacement_concentric = dataframe["milieu.1"].iloc[peak] - dataframe["milieu.1"].iloc[list_peak[peak_index]]
        vertical_displacement_concentric_meter = vertical_displacement_concentric * ratio
    
        time_vertical_displacement_concentric = list_peak[peak_index] - peak
        time_vertical_displacement_concentric_seconds = time_vertical_displacement_concentric / video_frequency
    
        vertical_movement_speed.append(vertical_displacement_concentric_meter / time_vertical_displacement_concentric_seconds)
    return vertical_movement_speed

def normalised_series(serie):
    original_index = np.linspace(0, len(serie) - 1, len(serie))
    normalised_index = np.linspace(0, len(serie) - 1, 101)
    normalised_data = np.interp(normalised_index, original_index, serie)
    return pd.Series(abs(normalised_data))  

def mean_sd_centric(serie, start, finish):
    bar_path=[]
    serie=serie-min(serie)
    for i in range(len(start)):
        bar_path_temp=normalised_series(serie[start[i]:finish[i]])
        bar_path.append(bar_path_temp)
    bar_path=np.array(bar_path)
    mean_bar_path=np.mean(bar_path,axis=0)
    sd_bar_path=np.std(bar_path,axis=0)
    return mean_bar_path, sd_bar_path

def filtre_passe_bas_serie(serie, frequence_coupure=6, ordre_filtre=4):
    frequence_normalisee = frequence_coupure / (0.5 * serie.index[-1])
    b, a = butter(ordre_filtre, frequence_normalisee, btype='low', analog=False, output='ba')
    serie_filtree = filtfilt(b, a, serie)
    return serie_filtree

def read_csv(csv_path):
    dataframe = load_df(csv_path)
    plate_size = calculate_plate_size(dataframe)
    ratio = calculate_ratio(plate_size)
    peaks = find_height_peak(dataframe["milieu.1"])
    list_start_peak, list_end_peak = find_start_end_peak(dataframe["milieu.1"], peaks)
    vertical_movement_speed = calculate_concentric_speed(dataframe, peaks, list_start_peak, list_end_peak, ratio)
    return vertical_movement_speed, dataframe, peaks, list_start_peak, list_end_peak, ratio

# User Interface
root = tk.Tk()
root.title("Video Processing Application")

# Global Variables
csv_path = ""
video_frequency = 29 # Hz
real_plate_size = 0.45  # meters


# User Interface Functions
def select_video():
    global original_video_path
    original_video_path = filedialog.askopenfilename(title="Select Original Video File")

def select_resized_video():
    global resized_video_path
    resized_video_path = filedialog.askopenfilename(title="Select Resized Video File")

def select_config_file():
    global config_file_path
    config_file_path = filedialog.askopenfilename(title="Select Config File")

def select_output_path():
    global output_path
    output_path = filedialog.asksaveasfilename(title="Save As")

def resize_video(input_file, output_file, width, height):
    cap = cv2.VideoCapture(input_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def analyse_video():
    deeplabcut.analyze_videos(config_file_path, resized_video_path, videotype="mp4", save_as_csv=True)
    deeplabcut.filterpredictions(config_file_path, resized_video_path)

def create_labeled_video():
    deeplabcut.create_labeled_video(config_file_path, resized_video_path, videotype="mp4", filtered=True, displayedbodyparts="milieu", trailpoints=30)

def select_csv_file():
    global csv_path
    csv_path = filedialog.askopenfilename(title="Select Paths CSV File")

def process_csv_file():
    if not csv_path:
        print("Please select a csv file first.")
        return

    vertical_speed, data, peaks, start, end, ratio = read_csv(csv_path)
    
    x=data["milieu"]*ratio
    y=data["milieu.1"]*ratio

    x=filtre_passe_bas_serie(x,frequence_coupure=50,ordre_filtre=2) 
    y=filtre_passe_bas_serie(y,frequence_coupure=50,ordre_filtre=2)

    mean_excentrics_x, sd_excentrics_x = mean_sd_centric(x,start,peaks)
    mean_excentrics_y, sd_excentrics_y = mean_sd_centric(y,start,peaks)

    mean_concentrics_x, sd_concentrics_x = mean_sd_centric(x,peaks,end)
    mean_concentrics_y, sd_concentrics_y = mean_sd_centric(y,peaks,end)

    # Create the figure
    fig = plt.Figure(figsize=(10, 6))

    # Average trajectory
    ax1 = fig.add_subplot(121)
    ax1.plot(mean_excentrics_x,mean_excentrics_y,linewidth=4, label="mean excentrics")
    ax1.plot(mean_concentrics_x,mean_concentrics_y,linewidth=4, label="mean concentrics")
    ax1.fill_between(mean_excentrics_x, (mean_excentrics_y - sd_excentrics_y), (mean_excentrics_y + sd_excentrics_y), color='gray', alpha=0.3, label='± SD')
    ax1.fill_between(mean_concentrics_x, (mean_concentrics_y - sd_concentrics_y), (mean_concentrics_y + sd_concentrics_y), color='gray', alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlabel("Horizontal displacement (meters)")
    ax1.set_ylabel("Vertical displacement (meters)")
    ax1.grid()
    ax1.legend()
    ax1.set_title("Average trajectory of the bar")

    # Vertical speed
    ax2 = fig.add_subplot(122)
    ax2.plot(np.array(list(range(len(vertical_speed)))) + 1, vertical_speed)
    ax2.grid()
    ax2.set_title("Vertical speed over reps of a serie")
    ax2.set_ylabel("Speed [m/s]")
    ax2.set_xlabel("Reps")

    # Display the figure in the application
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


# User Interface Buttons
select_video_button = tk.Button(root, text="Select Original Video", command=select_video)
select_video_button.pack()

select_output_button = tk.Button(root, text="Select Output Path", command=select_output_path)
select_output_button.pack()

resize_button = tk.Button(root, text="Resize Video", command=lambda: resize_video(original_video_path, output_path, 360, 640))
resize_button.pack()

select_resized_video_button = tk.Button(root, text="Select Resized Video", command=select_resized_video)
select_resized_video_button.pack()

select_config_button = tk.Button(root, text="Select Config File", command=select_config_file)
select_config_button.pack()

analyse_button = tk.Button(root, text="Analyze Resized Video", command=analyse_video)
analyse_button.pack()

create_button = tk.Button(root, text="Create Labeled Video", command=create_labeled_video)
create_button.pack()

# Add a button to select the "paths" file
select_paths_button = tk.Button(root, text="Select csv File", command=select_csv_file)
select_paths_button.pack()

# Add a button to process the selected "paths" file
process_paths_button = tk.Button(root, text="Process csv File", command=process_csv_file)
process_paths_button.pack()

# Launching the interface
root.mainloop()
