import cv2
import numpy as np
import pandas as pd
import ast

import os
# this file takes in the list of the image paths (tortuous vessel segments), as well as the original image (whole vessel network)
# it will output an RBG image with tortuous vessels colored red
# we will assume that we have the log_file path: "vessels_localized_log.csv"


def get_visualized_img(image_path, tortuous_list, is_prediction):
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
    # Iterate over the coordinates and set the pixel color to red
    for ind, row in final_df.iterrows():
        ys = ast.literal_eval(row[2])  # row[2] is a string
        xs = ast.literal_eval(row[1])
        all_x.extend(xs)
        all_y.extend(ys)
    all_x = np.array(all_x, dtype=int)
    all_y = np.array(all_y, dtype=int)


    # Set pixels to red (BGR = [0, 0, 255])
    image_rgb[all_y, all_x] = [0, 255, 0]

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



