import numpy as np
import cv2  
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from .params import min_peak_threshold, max_peak_threshold

def calculate_curvature(spline_x, spline_y, arc_length):
    """Improved curvature calculation with proper point matching"""
    """Directly use the original splines"""
    dx = spline_x.derivative(1)(arc_length)
    dy = spline_y.derivative(1)(arc_length)
    ddx = spline_x.derivative(2)(arc_length)
    ddy = spline_y.derivative(2)(arc_length)
    
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5 + 1e-6
    return (numerator / denominator)[1:-1]

def midpoint_deviation(coordinates):
    midpoint_idx = len(coordinates) // 2
    midpoint = coordinates[midpoint_idx]
    line_vector = coordinates[-1] - coordinates[0]
    return np.abs(np.cross(midpoint - coordinates[0], line_vector)) / np.linalg.norm(line_vector)

def straightness_index(coordinates):
    """Calculate straightness index with loop handling"""
    if len(coordinates) < 2:
        return 0.0
    
    # Vectorized length calculation
    diffs = np.diff(coordinates, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))

    # Endpoint distance
    start = np.array(coordinates[0])
    end = np.array(coordinates[-1])
    endpoint_distance = np.linalg.norm(end - start)
    
    if endpoint_distance < 1e-6:
        return np.inf  # Looped vessel
    
    return total_length / endpoint_distance, total_length

def bounding_box_aspect_ratio(coordinates):
    """Calculate aspect ratio of minimal bounding box"""
    coords = np.array(coordinates)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    width, height = max_vals - min_vals
    return width / height if height != 0 else 0.0

def count_significant_turns(curvatures, curvature_threshold=0.09, window_size=3):
    """Enhanced version that returns turn locations"""
    if not len(curvatures):
        return 0, 0.0, []

    turn_count = 0
    max_curvature = 0.0
    current_sign = 0
    consecutive_count = 0
    turn_indices = []

    # Find initial valid sign
    for idx, curv in enumerate(curvatures):
        if abs(curv) > curvature_threshold:
            current_sign = np.sign(curv)
            max_curvature = max(abs(curv), max_curvature)
            break

    for idx, curv in enumerate(curvatures):
        max_curvature = max(abs(curv), max_curvature)
        
        if abs(curv) < curvature_threshold:
            consecutive_count = 0
            continue
            
        if np.sign(curv) != current_sign:
            consecutive_count += 1
            if consecutive_count >= window_size:
                turn_count += 1
                turn_indices.append(idx - window_size + 1)  # Track first change position
                current_sign = np.sign(curv)
                consecutive_count = 0
        else:
            consecutive_count = 0

    return turn_count, max_curvature, turn_indices

def directional_entropy(coords, num_bins=8):
    angles = np.arctan2(np.diff(coords[:,1]), np.diff(coords[:,0]))
    hist = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))[0]
    prob = hist / hist.sum()
    return -np.sum(prob * np.log(prob + 1e-6))

def peak_counts(curvatures, max_peak_threshold=max_peak_threshold, min_peak_threshold=min_peak_threshold):
    smoothed = curvatures
    # Initialize containers
    positive_peaks = []
    negative_peaks = []
    for i in range(1, len(smoothed)-1):
        # Check for positive peak
        if (smoothed[i] > smoothed[i-1] and 
            smoothed[i] > smoothed[i+1] and 
            abs(smoothed[i]) >= min_peak_threshold):
            positive_peaks.append(i)
            
        # Check for negative peak
        elif (smoothed[i] < smoothed[i-1] and 
              smoothed[i] < smoothed[i+1] and 
              abs(smoothed[i]) >= min_peak_threshold):
            negative_peaks.append(i)
    return negative_peaks, positive_peaks, len(negative_peaks) + len(positive_peaks)

def extract_features(coordinates, raw_coords, smoothed):
    """Main feature extraction with proper normalization"""
    features = {}
    
    # Precompute vessel length once
    diffs = np.diff(coordinates, axis=0)
    vessel_length = np.sum(np.linalg.norm(diffs, axis=1))
    num_points = len(coordinates)
    
    # 1. Curvature features
    curvatures = calculate_curvature(smoothed["spline_x"], smoothed["spline_y"], smoothed["arc_length"])
    features["Curvature_List"] = curvatures
    
    turn_count, max_curv, turn_indices = count_significant_turns(curvatures)
    features.update({
        "Turn_Count": turn_count,
        "Max_Curvature": max_curv,
        "Turn_Locations": turn_indices
    })
    
    # 2. Total curvature (scale-invariant)
    avg_step = vessel_length / (num_points - 1) if num_points > 1 else 0
    features["Total_Curvature"] = np.sum(np.abs(curvatures)) * avg_step
    features["Mean Curvature"] = np.mean(np.abs(curvatures)) if len(curvatures) > 0 else 0
    # 3. Straightness index
    features["Straightness_Index"], total_length = straightness_index(coordinates)
    
    features["Turns-Length Ratio"] = features["Turn_Count"] / total_length if total_length > 0 else 0
    # 4. Aspect ratio
    features["Aspect_Ratio"] = bounding_box_aspect_ratio(coordinates)
    
    # 6. Curvature statistics
    features["Curvature_Variance"] = np.var(curvatures) 
    
    cov_matrix = np.cov(coordinates.T)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    features["PCA_Eigen_Ratio"] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0

    sign_changes = np.where(np.diff(np.sign(curvatures)))[0]
    features["Inflection_Points"] = len(sign_changes)
    # features["Bending Energy"] = np.sum(curvatures ** 2)
    features["Midpoint Deviation"] = midpoint_deviation(coordinates)
    features["Length of Vessel"] = total_length
    features["Directional Entropy"] = directional_entropy(coordinates)
    _, _, features["Peak Counts"] = peak_counts(curvatures)

    return features