import os
import cv2
import pandas as pd
import numpy as np
from .utils import find_path
from .features import extract_features
from .utils import find_path, smooth_and_resample
from tqdm import tqdm
import random

def process_vessel(image_path):
    # Load and binarize image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        raw_coords = find_path(binary)
        if raw_coords == -1:
            return None #skipped due to loop
        if len(raw_coords) < 2:
            print("Error: No vessel detected")
            return None
    except Exception as e:
        print(f"Path finding failed: {str(e)}")
        return None
    
    # Smooth and resample
    smoothed = smooth_and_resample(raw_coords)
    smoothed_coords = smoothed["points"]

    # Calculate features
    features = extract_features(smoothed_coords, raw_coords, smoothed)
    
    # # Visualization with turn locations
    # visualize_vessel(smoothed_coords, 
    #                 features['Curvature_List'],
    #                 features['Turn_Locations'])
    

    del features["Turn_Locations"] # we don't need this
    del features["Curvature_List"]
    return features


skipped = 0

def extract_features_for_dataset(image_list, dir_path):
    global skipped
    data = []
    for filename in tqdm(image_list, desc=f"Processing Images"):
        img_path = os.path.join(dir_path, filename)
        features = process_vessel(str(img_path))  # Extract vessel features
        if features is None:
            skipped += 1
            continue
        # features["label"] = label
        data.append(features)
    return data


def process_images(data_dir):
    images_tor = [f for f in os.listdir(os.path.join(data_dir, "tortuous")) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
    images_non = [f for f in os.listdir(os.path.join(data_dir, "non_tortuous")) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
    print("Total images found are", len(images_tor) + len(images_non))
    data_tor = extract_features_for_dataset(images_tor, os.path.join(data_dir, "tortuous"))
    data_non = extract_features_for_dataset(images_non, os.path.join(data_dir, "non_tortuous"))
    return pd.DataFrame(data_tor + data_non)
