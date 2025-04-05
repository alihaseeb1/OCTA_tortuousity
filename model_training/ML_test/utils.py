import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from .params import POINT_RESAMPLING_SIGMA,CUBIC_INTERPOLATION_POINTS

from .params import (
    CUBIC_INTERPOLATION_POINTS, 
    POINT_RESAMPLING_SIGMA
    )

def find_path(segment):
    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    neighbor_count = cv2.filter2D(segment, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere((neighbor_count == 1) & (segment == 1))

    if len(endpoints) != 2:
        # raise ValueError("Exactly two endpoints required for a single vessel segment.")
        # print("skipped becaues of loop structure")
        return -1
    start = tuple(endpoints[0])
    end = tuple(endpoints[1])

    path = [start]
    current = start
    prev = (-1, -1)  # Track previous point to avoid backtracking
    visited =set()
    while current != end:
        y, x = current
        visited.add(current)
        neighbors = []
        # Check all 8-connected neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue  # Skip current pixel
                ny, nx = y + dy, x + dx  # Correct: dy affects y, dx affects x
                # Bounds check and ensure it's part of the vessel
                if 0 <= ny < segment.shape[0] and 0 <= nx < segment.shape[1]:
                    if segment[ny, nx] == 1 and (ny, nx) != prev:
                        if (ny, nx) in visited:
                            continue
                        else:
                            neighbors.append((ny, nx))
        if not neighbors:
            path.pop(-1)
            current = path[-1]
            prev = path[-2] if len(path) > 1 else (-1, -1)
            continue
            # raise RuntimeError("Path tracing failed: No valid neighbors.")

        # Move to the next pixel (prioritize non-diagonal steps if needed)
        next_pixel = neighbors[0]
        path.append(next_pixel)
        prev = current
        current = next_pixel

    return path

def smooth_and_resample(coordinates, sigma=POINT_RESAMPLING_SIGMA, num_points=CUBIC_INTERPOLATION_POINTS):
    """Smooth coordinates and resample using spline interpolation"""
    if len(coordinates) < 4:
        # Handle very short paths
        print("Warning: Path too short for spline fitting")
        return coordinates
    
    # Convert to numpy array
    coords = np.array(coordinates)
    
    # 1. Gaussian smoothing
    smoothed_x = gaussian_filter1d(coords[:, 0].astype(float), sigma=sigma)
    smoothed_y = gaussian_filter1d(coords[:, 1].astype(float), sigma=sigma)
    
    # 2. Calculate cumulative arc length
    diff = np.diff(np.column_stack([smoothed_x, smoothed_y]), axis=0)
    arc_length = np.cumsum(np.linalg.norm(diff, axis=1))
    arc_length = np.insert(arc_length, 0, 0)
    
    # 3. Fit parametric cubic spline
    cs_x = CubicSpline(arc_length, smoothed_x)
    cs_y = CubicSpline(arc_length, smoothed_y)
    
    # 4. Resample at regular intervals
    new_arc_length = np.linspace(0, arc_length[-1], num_points)
    resampled_x = cs_x(new_arc_length)
    resampled_y = cs_y(new_arc_length)
    
    return {
        'points': np.column_stack([resampled_x, resampled_y]),
        'spline_x': cs_x,
        'spline_y': cs_y,
        'arc_length': new_arc_length
    }