import numpy as np
import cv2
import math
from ...model_training.ML_test.utils import find_path

def ti(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours of the vessel
    vessel_path = find_path(image)
    if vessel_path == -1:
        return 0
    # Calculate the arc length (L_c) by summing up distances between consecutive points
    L_c = 0
    for i in range(1, len(vessel_path)):
        x1, y1 = vessel_path[i-1]
        x2, y2 = vessel_path[i]
        L_c += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Calculate the chord length (L_x) between the first and last point
    x_start, y_start = vessel_path[0]
    x_end, y_end = vessel_path[-1]
    L_x = math.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    # print(f"L_x is {L_x} and L_c is {L_c}")
    # Calculate the tortuosity
    tortuosity = (L_c / L_x) - 1

    # print(f"Tortuosity: {tortuosity}")
    return tortuosity