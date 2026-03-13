# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:03:19 2025

@author: francois
"""

import deeplabcut
import pandas as pd
import numpy as np
import statistics
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from scipy.signal import butter, filtfilt
from sklearn import linear_model

# Definition of data processing functions

# Load the csv file created by deeplabcut
def load_df(csv_file_path):
    data = pd.read_csv(csv_file_path)
    if data.shape[1]==10 :
        dataframe = pd.read_csv(csv_file_path, header=1)
        dataframe = dataframe.drop(0)
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        dataframe = dataframe.astype(float)
        dataframe = dataframe.iloc[5:-5]
        dataframe = dataframe.reset_index(drop=True)
    # TODO : for future csv file made with the webcam
    else : 
        dataframe = pd.read_csv(csv_file_path, header=0)
    return dataframe

# Get the ratio bewteen pixels in the frame and the plate size to scale everything
def calculate_ratio(median_plate_size,real_plate_size=0.45):
    real_plate_size = 0.45
    ratio = real_plate_size / median_plate_size
    return ratio

# for webcam processing
def draw_tracker_point(frame, x, y, color=(0, 255, 0)):
    """Draw point on frame."""
    cv2.circle(frame, (x, y), 3, color, -1)
    
# for webcam processing
def get_average_position(positions_deque):
    """compute mean pose from deque."""
    if not positions_deque:
        return None
    avg_x = int(sum(p[0] for p in positions_deque) / len(positions_deque))
    avg_y = int(sum(p[1] for p in positions_deque) / len(positions_deque))
    return (avg_x, avg_y)

# Calculate the distance in pixels between the top and the bottom of the plate
def calculate_plate_size(dataframe):
    x2, x1 = dataframe["bas"], dataframe["haut"]
    y2, y1 = dataframe["bas.1"], dataframe["haut.1"]
    plate_size = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    median_plate_size = statistics.median(plate_size) # Median in case of outlier detection
    return median_plate_size

# kinda useless
def compute_speed(y,dt):
    return np.gradient(y, dt)

# Normalize on 101 points to be comparable
def normalize_rep(rep_signal, n_points=101):
    x_old = np.linspace(0, 1, len(rep_signal))
    x_new = np.linspace(0, 1, n_points)

    f = interp1d(x_old, rep_signal, kind="linear")
    return f(x_new)

# Normalize for x and y on 101 points to be comparable
def normalize_2d(x, y, n_points=101):
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_points)

    fx = interp1d(t_old, x, kind="linear")
    fy = interp1d(t_old, y, kind="linear")

    return fx(t_new), fy(t_new)

# Cutoff = 8 based on previously made sensitivity analysis 
def lowpass_butterworth_zero_lag(data, fs, cutoff=8, order=4):
    """
    data : array 1D (vertical pose)
    fs : Acquisition Frequency (Hz)
    cutoff : Cutting frequency (Hz)
    order : filter order
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)  # zero-lag
    
    return filtered

def detect_reps_hybrid_metric_peaks(
    y_filt,
    fs=29,
    cutoff=8,
    peak_distance=30, 
    peak_prominence=0.15, # At least 0.15 m rom to be detected
    vel_thresh=0.08,  # plus haut que 0 car tracking bruité + 30 fps
    low_zone=0.40,
    high_zone=0.60,
    max_search_seconds=4,
    exercice_type="squat_bench_like"
):
    """
    Hybrid approach:
    - peaks = top of each rep
    - start/end = velocity threshold + position constraints (MetricVBT style)
    """
    
    #y = np.asarray(y)
    if exercice_type == "squat_bench_like" : 
        y_filt=y_filt 
    else : 
        y_filt=-y_filt
        

    # Filter position
    # y_filt = lowpass_butterworth_zero_lag(y, fs, cutoff=cutoff, order=4)

    # Velocity
    dt = 1 / fs
    v = compute_speed(y_filt, dt)
    v = lowpass_butterworth_zero_lag(v, fs, cutoff=cutoff, order=4)
    v = abs(v)
    
    # Find peaks (top positions)
    peaks, _ = find_peaks(y_filt, distance=fs, prominence=peak_prominence,width=10)

    # Compute ROM bounds
    y_min = np.min(y_filt)
    y_max = np.max(y_filt)
    ROM = y_max - y_min

    low_threshold_pos = y_min + low_zone * ROM
    high_threshold_pos = y_min + high_zone * ROM

    starts = []
    ends = []

    max_search_frames = int(max_search_seconds * fs)
    
    for peak in peaks :
        start_search_begin = max(0, peak - max_search_frames)

        for i in range(peak, start_search_begin, -1):
            if (v[i] < vel_thresh) and (y_filt[i] < low_threshold_pos):
                starts.append(i)
                break
            
        end_search_end = min(len(y_filt)-1, peak + max_search_frames)

        for i in range(peak, end_search_end):
             # Metric-like end: velocity drops + high position zone
            if (v[i] < vel_thresh) and (y_filt[i] < low_threshold_pos):
                 ends.append(i)
                 break
             
    # In case no start was found for the first rep if the video was misscut
    # Very specific, but has actually happened to me. Could still drag it by hand later
    if len(starts)<len(peaks):
        starts.insert(0,0)

    return starts, ends, list(peaks)

# TODO : add fs 
def determine_exercise_type(
    y_filt,
    peak_distance=30,
    peak_prominence=0.15
    ):
    
    tops, _ = find_peaks(y_filt, distance=peak_distance, prominence=peak_prominence,width=10)
    bottoms, _ = find_peaks(-y_filt, distance=peak_distance, prominence=peak_prominence,width=10)


    if len(bottoms) == 0 :
        mode="squat_bench_like"
        return mode
    if len(tops) == 0:
        mode ="deadlift_like"
        return mode

    # First turning point
    if bottoms[0] < tops[0]:
        mode = "deadlift_like"   # bottom -> top -> bottom
    else:
        mode = "squat_bench_like"  # top -> bottom -> top
        
    return mode
        
# To get a good concentric start point even if user is doing 3ct or similar
# TODO : change this black magic shenanigan  
def find_start_concentric(y_filt, y_speed_filt,peaks,fs=30):
    start_concentric=[]
    max_search_seconds = 3
    vel_thresh = 0.02
    max_search_frames = int(max_search_seconds * fs)
    
    for peak in peaks :
        end_search = max(0, peak + max_search_frames)
        for i in range(peak, end_search):
            if (abs(y_speed_filt[i]) > vel_thresh) and (y_filt[i]>np.mean(y_filt[i:i+8])+np.mean(y_filt[i:i+8])*0.01):
                start_concentric.append(i)
                break    
    return start_concentric

def ROM(result):
    reps_y = result["reps_y"]
    roms = []
    for i in range(len(reps_y)):
        rom = np.max(reps_y[i,:])-np.min(reps_y[i,:])
        roms.append(rom)
    return roms

def compute_ROM_across_csv(results):
    average_roms = []
    for i in range(len(results)):
        roms = ROM(results[i])
        average_roms.append(np.mean(np.array(roms)))
    return average_roms

def read_analyse_csv_routine(
    csv,
    video_frequency=30,
    real_plate_size=0.45,
    fc=8,
    verbatim=1
    ):
    """
    csv : deeplabcut csv file
    video_frequency : fréquence d'échantillonnage (Hz)
    fc : fréquence de coupure (Hz)
    real_plate_size : size of a plate (m)
    verbatim : to print usefull info, helps debugging

    """
    
    # Load the dataframe, calculate plate size, and ratio between pixel in the frame and real_plate_size
    dataframe = load_df(csv)
    plate_size = calculate_plate_size(dataframe)
    ratio = calculate_ratio(plate_size,real_plate_size=real_plate_size)
    
    # Convert pixel into meters
    x=dataframe["milieu"]*ratio
    y=dataframe["milieu.1"]*ratio
    # Filter with a low pass butterworth zero lag order 4, FC = 8 determined with residual analysis
    # MetricVBT apparently uses fc = 10, but my model is a little worse I believe stronger filter is needed
    x_filt=lowpass_butterworth_zero_lag(x,video_frequency,fc,4) 
    y_filt=lowpass_butterworth_zero_lag(y,video_frequency,fc,4)
    
    # Determine wich exercise type the csv is (squat/bench or deadlift type)
    # To know what part of the exercise is the concentric part
    mode = determine_exercise_type(y_filt)
    if verbatim == 1 :
        print(mode)
    # Find the reps in the time serie based on the height signal
    # Algorythme : speed < 0.05 in the 3s before/after a peak to determine start and end of rep
    # peaks defined as at least >20cm differences in height
    start, end,peaks = detect_reps_hybrid_metric_peaks(y_filt, exercice_type=mode, fs=video_frequency)
    
    # To get similar values between lift types
    if mode == "squat_bench_like" : 
        y_filt = y_filt
        y_speed_filt = compute_speed(y_filt,1/video_frequency)
        y_speed_filt_filt = lowpass_butterworth_zero_lag(y_speed_filt,video_frequency,fc,4)
        start_concentric = find_start_concentric(y_filt,y_speed_filt_filt,peaks,fs = video_frequency)


    else : 
        y_filt = y_filt[0]-y_filt
        y_speed_filt = compute_speed(y_filt,1/video_frequency)
        y_speed_filt_filt = lowpass_butterworth_zero_lag(y_speed_filt,video_frequency,fc,4)

    
    # Get mean and max concentric speed of each rep of the csv
    mean_speed=[]
    max_speed=[]
    for i in range(len(peaks)) : 
        if mode == "squat_bench_like" : 
            concentric_speed = abs(y_speed_filt_filt[start_concentric[i]:end[i]])
        else : 
            concentric_speed = abs(y_speed_filt_filt[start[i]:peaks[i]])
        mean_speed.append(np.mean(concentric_speed))
        max_speed.append(np.max(concentric_speed))
    
    # normalizing reps on 100% to calculate mean trajectory
    normalized_reps_y=[]
    normalized_reps_x=[]
    for i in range(len(peaks)):
        repetition_y = y_filt[start[i]:end[i]]
        repetition_x = x_filt[start[i]:end[i]]

        normalized_rep_x,normalized_rep_y = normalize_2d(repetition_x,repetition_y, n_points=101)
        normalized_reps_y.append(normalized_rep_y)
        normalized_reps_x.append(normalized_rep_x)

    normalized_reps_x = np.array(normalized_reps_x)
    normalized_reps_y = np.array(normalized_reps_y)

    mean_traj_y = np.mean(normalized_reps_y, axis=0)
    sd_traj_y = np.std(normalized_reps_y, axis=0)

    mean_traj_x = np.mean(normalized_reps_x, axis=0)
    sd_traj_x = np.std(normalized_reps_x, axis=0)
    
    result = {
            "y_filt": y_filt,
            "x_filt": x_filt,
            "y_speed": y_speed_filt_filt,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "reps_x": normalized_reps_x,
            "reps_y": normalized_reps_y,
            "mean_traj_x": mean_traj_x,
            "mean_traj_y": mean_traj_y,
            "sd_traj_x": sd_traj_x,
            "sd_traj_y": sd_traj_y,
            "mode": mode,
            "start":start,
            "end":end,
            "peaks":peaks,
        }
    if "start_concentric" in locals():
        result["start_concentric"] = start_concentric
    return result

def update_result(result, new_starts, new_peaks, new_ends, new_start_concentric=None):     
    # Get mean and max concentric speed of each rep of the csv
    mean_speed=[]
    max_speed=[]
    for i in range(len(new_peaks)) : 
        if new_start_concentric != None : 
            concentric_speed = abs(result["y_speed"][new_start_concentric[i]:new_ends[i]])
        else : 
            concentric_speed = abs(result["y_speed"][new_starts[i]:new_peaks[i]])
        mean_speed.append(np.mean(concentric_speed))
        max_speed.append(np.max(concentric_speed))
    
    # normalizing reps on 100% to calculate mean trajectory
    normalized_reps_y=[]
    normalized_reps_x=[]
    for i in range(len(new_peaks)):
        repetition_y = result["y_filt"][new_starts[i]:new_ends[i]]
        repetition_x = result["x_filt"][new_starts[i]:new_ends[i]]

        normalized_rep_x,normalized_rep_y = normalize_2d(repetition_x,repetition_y, n_points=101)
        normalized_reps_y.append(normalized_rep_y)
        normalized_reps_x.append(normalized_rep_x)

    normalized_reps_x = np.array(normalized_reps_x)
    normalized_reps_y = np.array(normalized_reps_y)

    mean_traj_y = np.mean(normalized_reps_y, axis=0)
    sd_traj_y = np.std(normalized_reps_y, axis=0)

    mean_traj_x = np.mean(normalized_reps_x, axis=0)
    sd_traj_x = np.std(normalized_reps_x, axis=0)
    
    result["mean_speed"] = mean_speed
    result["max_speed"] = max_speed
    result["reps_x"] = normalized_reps_x
    result["reps_y"] = normalized_reps_y
    result["mean_traj_x"]= mean_traj_x
    result["mean_traj_y"]= mean_traj_y
    result["sd_traj_x"]= sd_traj_x
    result["sd_traj_y"]= sd_traj_y
    
    return result

def compute_load_velocity_profil(results,li_kg):
    
    rep_speed_of_different_series=[]
    for i in range(len(li_kg)):
        print(i)
        rep_speed_of_different_series.append(np.max(results[i]["mean_speed"]))
    rep_speed_of_different_series=np.array(rep_speed_of_different_series)    
    
    # linear regr model training
    loads = li_kg.reshape(len(li_kg),1) 
    mean_speeds = rep_speed_of_different_series.reshape(len(li_kg),1)
    regr = linear_model.LinearRegression()
    regr.fit(mean_speeds, loads) 

    load_velocity_profil = {
        "mean_speeds":mean_speeds,
        "regr":regr
        }
    return load_velocity_profil


# Plot functions to help debugging
def plot_mean_max_speed(results):
    max_speed = results["max_speed"]
    mean_speed = results["mean_speed"]
    reps=np.arange(len(max_speed))+1
    if len(reps)==1 :
        bar_labels = ["max_speed","mean_speed"]
        plt.figure()
        plt.bar([1,2],[max_speed[0],mean_speed[0]],tick_label=bar_labels)
        plt.title("Vitesse de la rep")
        plt.ylabel("Vitesse (m/s)")
        plt.show()
    else :
        plt.figure()
        plt.plot(reps,np.array(max_speed),label="max_speed")
        plt.plot(reps,np.array(mean_speed),label="mean_speed")
        plt.grid()
        plt.title("Vitesse fonction des reps de la série")
        plt.xlabel("reps")
        plt.ylabel("Vitesse (m/s)")
        plt.legend()
        plt.show()
        

def plot_trajectory(results):
    normalized_reps_y = results["reps_y"]
    normalized_reps_x = results["reps_x"]
    mean_traj_y = results["mean_traj_y"]
    mean_traj_x = results["mean_traj_x"]
    sd_traj_y = results["sd_traj_y"]
    sd_traj_x = results["sd_traj_x"]
    mode = results["mode"]
    
    plt.figure()
    for i in range(len(results["max_speed"])):
        plt.plot(normalized_reps_y[i,:], normalized_reps_x[i,:], alpha=0.2)
    plt.plot(mean_traj_y, mean_traj_x)
    #plt.fill_betweenx(mean_traj_y, mean_traj_x - sd_traj_x, mean_traj_x + sd_traj_x, alpha=0.2, label="± SD (x)")
    plt.fill_betweenx(mean_traj_x, mean_traj_y - sd_traj_y, mean_traj_y + sd_traj_y, alpha=0.2, label="± SD (x)")
    
    if mode == "squat_bench_like" :
        plt.gca().invert_yaxis()  # optionnel si coordonnées image (y vers le bas)    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectoire barre (rep)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()