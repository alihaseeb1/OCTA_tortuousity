import cv2
import numpy as np
import pandas as pd
import ast

import os
# this file takes in the list of the image paths (tortuous vessel segments), as well as the original image (whole vessel network)
# it will output an RBG image with tortuous vessels colored red
# we will assume that we have the log_file path: "vessels_localized_log.csv"
color_list = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255)  # Magenta
]

def get_vessel_color(vessel_name):
    """
    Assigns color based on the file name. If the file name starts with a number or prefix,
    that will determine the index in the color list.
    We use this to make sure that we get the same color for each vessel every time

    Args:
        vessel_name (str): The vessel file name.

    Returns:
        tuple: A BGR color tuple.
    """
    prefix = vessel_name.split('_')[0]  # Example: Extract '123' from '123_vessel_name.png'

    try:
        index = int(prefix) % len(color_list)  # Ensure the index is within the color list bounds
    except ValueError:
        index = hash(vessel_name) % len(color_list)  # Using hash to get a pseudo-random index

    return color_list[index]

def get_visualized_img(image_path, tortuous_list, is_prediction, region_size=1):
    """
    Visualizes tortuous vessel segments on an image.

    Args:
        image_path (str): Path to the original grayscale image.
        tortuous_list (list): List of tortuous vessel segment file names.
        is_prediction (bool): Whether the visualization is for predictions or ground truth.

    Saves:
        Annotated RGB image in the "result" folder.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If coordinates in the CSV file cannot be parsed.
    Returns:
        [list of x-coordinates, list of y-coordinates] of the pixels of the tortuous vessels, total pixels that are part of the tortuous vessel
    """
    image_file = image_path.split("\\")[-1]

    # extract the file names
    tortuous_list = [os.path.basename(i) for i in tortuous_list]

    # Load the grayscale image (assuming it's a single-channel image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Load the CSV file containing the coordinates
    csv_path = os.path.join("model_eval/images", image_file.split(".")[0], 'vessels_localized_log.csv')  
    # They are in the format img file name, x coordinates, y coordinates
    coords_df = pd.read_csv(csv_path, header=None)

    mask = coords_df[0].isin(tortuous_list)

    final_df = coords_df[mask]

    if final_df.empty:
        print(f"No matching coordinates found in CSV for {image_file}.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    all_x = []
    all_y = []

    vessels = []
    # Iterate over the coordinates and color the vessels
    for ind, row in final_df.iterrows():
        vessel_name = row[0]
        ys = ast.literal_eval(row[2])  # row[2] is a string
        xs = ast.literal_eval(row[1])
        vessels.append((vessel_name, ys, xs, len(xs)))
    # all_x = np.array(all_x, dtype=int)
    # all_y = np.array(all_y, dtype=int)
    # sort the vessels so that we can color the shorter ones first and longer ones last (we can keep the longer one when they overlap)
    vessels.sort(key = lambda x: x[3])
    
    for vessel_name, ys, xs, _ in vessels:
        vessel_color = get_vessel_color(vessel_name)
        for x, y in zip(xs, ys):
            for dx in range(-region_size, region_size + 1):
                for dy in range(-region_size, region_size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:  # Ensure the pixel is within the image bounds
                        image_rgb[ny, nx] = vessel_color
        all_x.extend(xs)
        all_y.extend(ys)
    # Set pixels to red (BGR = [0, 0, 255])
    # image_rgb[all_y, all_x] = [0, 0, 255]

    # Save image
    if is_prediction:
        cv2.imwrite(os.path.join("model_eval/result", "predicted_"+ image_file.split(".")[0] + "_annotated" + '.png'), image_rgb)  # Save the image with red-colored pixels
    else:
        cv2.imwrite(os.path.join("model_eval/result", image_file.split(".")[0] + "_annotated" +  '.png'), image_rgb)  # Save the image with red-colored pixels
    
    return list(zip(all_x, all_y)), len(all_x)
    # to display
    # cv2.imshow('Image with Red Coordinates', image_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



