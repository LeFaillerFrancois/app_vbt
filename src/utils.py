# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:03:19 2025

@author: francois
"""

#import deeplabcut
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
def calculate_ratio(median_plate_size, real_plate_size=0.45):
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
    # Check the columns names (if calisthenics model is used)
    suffix = "bas" if "bas" in dataframe.columns else "wrist"
    prefix = "haut" if "bas" in dataframe.columns else "elbow"
    
    x2, y2 = dataframe[suffix], dataframe[f"{suffix}.1"]
    x1, y1 = dataframe[prefix], dataframe[f"{prefix}.1"]
    
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

# TODO : remake this fonction, work as intended but weird way to code it
def ROM(reps_y, result = None):
    if result :
        reps_y = result["reps_y"]
    roms = []
    for i in range(len(reps_y)):
        rom = np.max(reps_y[i,:])-np.min(reps_y[i,:])
        roms.append(rom)
    return roms

def compute_ROM_across_csv(results):
    average_roms = []
    reps_y = results[0]["reps_y"]
    for i in range(len(results)):
        roms = ROM(reps_y, results[i])
        average_roms.append(np.mean(np.array(roms)))
    return average_roms

def compute_metrics_from_indices(x_filt, y_filt, y_speed, starts, ends, peaks, mode, start_concentric=None):
    """Compute speeds and trajectories from indexs."""
    mean_speed = []
    max_speed = []
    
    for i in range(len(peaks)):
        if mode == "squat_bench_like" and start_concentric is not None:
            slice_idx = slice(start_concentric[i], ends[i])
        else:
            slice_idx = slice(starts[i], peaks[i])
            
        concentric_speed = abs(y_speed[slice_idx])
        mean_speed.append(np.mean(concentric_speed))
        max_speed.append(np.max(concentric_speed))
    
    # Normalisation
    reps_x, reps_y = [], []
    for i in range(len(peaks)):
        nx, ny = normalize_2d(x_filt[starts[i]:ends[i]], y_filt[starts[i]:ends[i]], n_points=101)
        reps_x.append(nx)
        reps_y.append(ny)

    reps_x, reps_y = np.array(reps_x), np.array(reps_y)
    roms = np.array(ROM(reps_y))
    
    return {
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "reps_x": reps_x,
        "reps_y": reps_y,
        "mean_traj_x": np.mean(reps_x, axis=0),
        "mean_traj_y": np.mean(reps_y, axis=0),
        "sd_traj_x": np.std(reps_x, axis=0),
        "sd_traj_y": np.std(reps_y, axis=0),
        "ROM":roms
    }

def preprocess_vbt_data(csv_path, video_frequency=30, real_plate_size=0.45, fc=8):
    df = load_df(csv_path)
    px_size = calculate_plate_size(df)
    # If height was given (>1m), use the forearm size from an estimation of Winter anthropometric tables
    if real_plate_size > 1:
        upper_limb_percentage = 0.16
        height = real_plate_size
        forearm_size = upper_limb_percentage * height
        forearm_size_px = px_size
        ratio = calculate_ratio(forearm_size_px,real_plate_size=forearm_size)
    else:
        ratio = calculate_ratio(px_size, real_plate_size=real_plate_size)

    suffix = "milieu" if "bas" in df.columns else "shoulder"
    x_raw, y_raw = df[suffix] * ratio, df[f"{suffix}.1"] * ratio
    
    x_filt = lowpass_butterworth_zero_lag(x_raw, video_frequency, fc, 4)
    y_filt = lowpass_butterworth_zero_lag(y_raw, video_frequency, fc, 4)
    
    return x_filt, y_filt, ratio

def read_analyse_csv_routine(csv, video_frequency=30, real_plate_size=0.45, fc=8):
    # 1. preprocess
    x_filt, y_filt, _ = preprocess_vbt_data(csv, video_frequency, real_plate_size, fc)
    mode = determine_exercise_type(y_filt)
    
    # 2. automatic rep detection
    start, end, peaks = detect_reps_hybrid_metric_peaks(y_filt, exercice_type=mode, fs=video_frequency)
    
    if mode != "squat_bench_like":
        y_filt_proc = y_filt[0] - y_filt
    else:
        y_filt_proc = y_filt
        
    y_speed = lowpass_butterworth_zero_lag(compute_speed(y_filt_proc, 1/video_frequency), video_frequency, fc, 4)
    
    sc = None
    if mode == "squat_bench_like":
        sc = find_start_concentric(y_filt_proc, y_speed, peaks, fs=video_frequency)

    # 3. Final computation
    results = compute_metrics_from_indices(x_filt, y_filt_proc, y_speed, start, end, peaks, mode, sc)
    
    # Adding new data to dic
    results.update({"y_filt": y_filt_proc, "x_filt": x_filt, "y_speed": y_speed, "mode": mode, 
                    "start": start, "end": end, "peaks": peaks, "csv_path": csv})
    if sc: results["start_concentric"] = sc
    return results

def update_result(result, new_starts, new_peaks, new_ends, new_start_concentric=None, fps=30, real_plate_size=0.45):
    # preprocess 
    x_filt, y_filt, ratio = preprocess_vbt_data(result["csv_path"], fps, real_plate_size)
    
    mode = result["mode"]
    y_filt_proc = (y_filt[0] - y_filt) if mode != "squat_bench_like" else y_filt
    y_speed = lowpass_butterworth_zero_lag(compute_speed(y_filt_proc, 1/fps), fps, 8, 4)
    
    # computation using new index
    metrics = compute_metrics_from_indices(x_filt, y_filt_proc, y_speed, new_starts, new_ends, new_peaks, mode, new_start_concentric)
    
    # Update the existing dic
    result.update(metrics)
    result.update({"y_filt": y_filt_proc, "x_filt": x_filt, "y_speed": y_speed, 
                    "start": new_starts, "end": new_ends, "peaks": new_peaks})
    if new_start_concentric: result["start_concentric"] = new_start_concentric
    
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