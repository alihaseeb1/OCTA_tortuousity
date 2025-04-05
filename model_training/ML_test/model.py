import os
import cv2
import pandas as pd
import numpy as np
from utils import find_path
from features import extract_features
from utils import find_path, smooth_and_resample
from visualize import visualize_vessel
from params import tortuous_path, non_tortuous_path
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

def extract_features_for_dataset(image_list, label, dir_path):
    global skipped
    data = []
    for filename in tqdm(image_list, desc=f"Processing {'Tortuous' if label == 1 else 'Non-Tortuous'}"):
        img_path = os.path.join(dir_path, filename)
        features = process_vessel(str(img_path))  # Extract vessel features
        if features is None:
            skipped += 1
            continue
        features["label"] = label
        data.append(features)
    return data



# Define dataset paths
data_dirs = {
    1: tortuous_path,
    0: non_tortuous_path
    }
balanced = False

# Load filenames
tortuous_images = [f for f in os.listdir(data_dirs[1]) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
non_tortuous_images = [f for f in os.listdir(data_dirs[0]) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]

# Shuffle to ensure randomness
random.shuffle(tortuous_images)
random.shuffle(non_tortuous_images)

if balanced:
    # Split tortuous images
    train_tortuous = tortuous_images[:int(len(tortuous_images) * 0.8)]
    val_tortuous = tortuous_images[int(len(tortuous_images) * 0.8):]

    # Match count for non-tortuous images
    train_non_tortuous = non_tortuous_images[:len(train_tortuous)]

    # Ensure validation set maintains the actual ratio
    non_to_tortuous_ratio = len(non_tortuous_images) / len(tortuous_images)
    val_non_tortuous_count = round(non_to_tortuous_ratio * len(val_tortuous))
    print(f"Total validation size of nontortuous while keeping correct ratio is {val_non_tortuous_count}")
    val_non_tortuous = random.sample(non_tortuous_images[len(train_tortuous):], val_non_tortuous_count)

    # Extract features for training and validation sets
    train_data = extract_features_for_dataset(train_tortuous, 1, data_dirs[1]) + extract_features_for_dataset(train_non_tortuous, 0, data_dirs[0])
    val_data = extract_features_for_dataset(val_tortuous, 1, data_dirs[1]) + extract_features_for_dataset(val_non_tortuous, 0, data_dirs[0])

    # Convert to DataFrame and save
    pd.DataFrame(train_data).to_csv("train_vessel_data.csv", index=False)
    pd.DataFrame(val_data).to_csv("val_vessel_data.csv", index=False)

    print("Training and validation datasets saved successfully!")
else:
    print(f"Total tortuous size of {len(tortuous_images)}")
    print(f"Total non-tortuous size of {len(non_tortuous_images)}")
    data = extract_features_for_dataset(tortuous_images, 1, data_dirs[1]) + extract_features_for_dataset(non_tortuous_images, 0, data_dirs[0])
    pd.DataFrame(data).to_csv("full_vessel_data.csv", index=False)


print(f"total skipped {skipped}")
