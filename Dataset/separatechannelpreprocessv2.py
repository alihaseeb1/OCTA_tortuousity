import os
import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu, gaussian
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects

# Preprocessing functions
def remove_gaps_in_vessels(binary_vessels, kernel_size_close=3, kernel_size_dilate=2, dilate_iterations=1):

    # Define a kernel (structuring element) for morphological operations
    kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)  # Larger kernel for closing
    kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)  # Smaller kernel for dilation
    
    # Perform morphological closing (dilation followed by erosion)
    closed_vessels = cv2.morphologyEx(binary_vessels, cv2.MORPH_CLOSE, kernel_close)
    
    # Perform dilation with a smaller kernel to connect vessels
    dilated_vessels = cv2.dilate(closed_vessels, kernel_dilate, iterations=dilate_iterations)
    
    return dilated_vessels

def load_and_remove_lines(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Replace the horizontal and vertical lines at specific positions
    gray_image[510, :] = gray_image[509, :]  # Replacing row 511 with row 510
    gray_image[511, :] = gray_image[512, :]  # Replacing row 512 with row 513
    gray_image[:, 510] = gray_image[:, 509]  # Replacing column 511 with column 510
    gray_image[:, 511] = gray_image[:, 512]  # Replacing column 512 with column 513
    
    # Adjust the contrast of the image (similar to `imadjust` in MATLAB)
    adjusted_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return adjusted_image

def remove_text(image):
    # Define the coordinates for the region containing the text
    upper_left = (806, 968)
    bottom_right = (1006, 996)
    
    # Replace the text region with white (or black, or surrounding color)
    image[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = 0  # White background
    
    return image

def apply_gaussian_filter(image, sigma):
    return gaussian(image, sigma=sigma, mode='reflect')

def apply_unsharp_mask(image, amount=1.5, radius=1):
    blurred = gaussian(image, sigma=radius)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0, 1)

def apply_clahe(image, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
    return clahe_image

def apply_frangi_filter(image, sigmas):
    frangi_response = frangi(image, sigmas=sigmas, black_ridges=False)
    return rescale_intensity(frangi_response)

def binarize_image(image, offset=0):
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value + offset
    return binary_image

def remove_noise(binary_image, min_size):
    cleaned_image = remove_small_objects(binary_image, min_size=min_size)
    return cleaned_image

def separate_vessel_sizes(image, small_range, medium_range, large_range,
                          min_size_small=50, min_size_medium=100, min_size_large=150,
                          gaussian_sigma_small=1, gaussian_sigma_medium=2, gaussian_sigma_large=3,
                          unsharp_amount_small=1.5, unsharp_amount_medium=1.5, unsharp_amount_large=1.5,
                          threshold_offset_small=0, threshold_offset_medium=0, threshold_offset_large=0,
                          clahe_params_before_small=(3.0, (8, 8)), clahe_params_before_medium=(3.0, (8, 8)), clahe_params_before_large=(3.0, (8, 8)),
                          clahe_params_after_small=(3.0, (8, 8)), clahe_params_after_medium=(3.0, (8, 8)), clahe_params_after_large=(3.0, (8, 8))):

    # Step 1: Smoothening
    smoothed_small = apply_gaussian_filter(image, sigma=gaussian_sigma_small)
    smoothed_medium = apply_gaussian_filter(image, sigma=gaussian_sigma_medium)
    smoothed_large = apply_gaussian_filter(image, sigma=gaussian_sigma_large)

    # Step 2: Sharpening
    sharpened_small = apply_unsharp_mask(smoothed_small, amount=unsharp_amount_small)
    sharpened_medium = apply_unsharp_mask(smoothed_medium, amount=unsharp_amount_medium)
    sharpened_large = apply_unsharp_mask(smoothed_large, amount=unsharp_amount_large)

    # Step 3: CLAHE before Frangi
    clahe_small_before = apply_clahe(sharpened_small, clip_limit=clahe_params_before_small[0], tile_grid_size=clahe_params_before_small[1])
    clahe_medium_before = apply_clahe(sharpened_medium, clip_limit=clahe_params_before_medium[0], tile_grid_size=clahe_params_before_medium[1])
    clahe_large_before = apply_clahe(sharpened_large, clip_limit=clahe_params_before_large[0], tile_grid_size=clahe_params_before_large[1])

    # Step 4: Frangi filter
    small_vessels = apply_frangi_filter(clahe_small_before, small_range)
    medium_vessels = apply_frangi_filter(clahe_medium_before, medium_range)
    large_vessels = apply_frangi_filter(clahe_large_before, large_range)

    # Step 5: CLAHE after Frangi
    clahe_small_after = apply_clahe(small_vessels, clip_limit=clahe_params_after_small[0], tile_grid_size=clahe_params_after_small[1])
    clahe_medium_after = apply_clahe(medium_vessels, clip_limit=clahe_params_after_medium[0], tile_grid_size=clahe_params_after_medium[1])
    clahe_large_after = apply_clahe(large_vessels, clip_limit=clahe_params_after_large[0], tile_grid_size=clahe_params_after_large[1])

    # Step 6: Binarization
    small_vessels_binary = binarize_image(clahe_small_after, threshold_offset_small)
    medium_vessels_binary = binarize_image(clahe_medium_after, threshold_offset_medium)
    large_vessels_binary = binarize_image(clahe_large_after, threshold_offset_large)

    # Step 7: Cleaning
    small_vessels_cleaned = remove_noise(small_vessels_binary, min_size=min_size_small)
    medium_vessels_cleaned = remove_noise(medium_vessels_binary, min_size=min_size_medium)
    large_vessels_cleaned = remove_noise(large_vessels_binary, min_size=min_size_large)

    return small_vessels_cleaned, medium_vessels_cleaned, large_vessels_cleaned

def combine_vessel_channels(small_vessels, medium_vessels, large_vessels,
                            small_weight=1.0, medium_weight=1.0, large_weight=1.0):
    small_vessels_normalized = small_vessels / np.max(small_vessels) if np.max(small_vessels) > 0 else small_vessels
    medium_vessels_normalized = medium_vessels / np.max(medium_vessels) if np.max(medium_vessels) > 0 else medium_vessels
    large_vessels_normalized = large_vessels / np.max(large_vessels) if np.max(large_vessels) > 0 else large_vessels

    combined_vessels = (small_weight * small_vessels_normalized +
                        medium_weight * medium_vessels_normalized +
                        large_weight * large_vessels_normalized)

    combined_vessels = combined_vessels / np.max(combined_vessels) if np.max(combined_vessels) > 0 else combined_vessels

    return combined_vessels

def binarize_combined_vessels(combined_vessels):
    threshold_value = threshold_otsu(combined_vessels)
    binary_vessels = combined_vessels > threshold_value
    return binary_vessels

def remove_noise(binary_vessels, min_size=100):
    cleaned_vessels = remove_small_objects(binary_vessels, min_size=min_size)
    return cleaned_vessels

def process_and_save_images(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            image = load_and_remove_lines(image_path=image_path)
            image = remove_text(image)

            # Define vessel size ranges for small, medium, and large vessels
            small_range = np.arange(1, 3)
            medium_range = np.arange(3, 6)
            large_range = np.arange(6, 10)

            # Apply the updated pipeline to separate vessels by size
            small_vessels_cleaned, medium_vessels_cleaned, large_vessels_cleaned = separate_vessel_sizes(
                image, small_range, medium_range, large_range,
                min_size_small=25, min_size_medium=50, min_size_large=70,
                gaussian_sigma_small=0.4, gaussian_sigma_medium=0.7, gaussian_sigma_large=0.2,
                unsharp_amount_small=2, unsharp_amount_medium=2, unsharp_amount_large=2,
                threshold_offset_small=0.1, threshold_offset_medium=0, threshold_offset_large=-0.05,
                clahe_params_before_small=(2.5, (8, 8)), clahe_params_before_medium=(0.5, (8, 8)), clahe_params_before_large=(0.1, (8, 8)),
                clahe_params_after_small=(4.0, (4, 4)), clahe_params_after_medium=(2.0, (16, 16)), clahe_params_after_large=(.1, (8, 8))
            )

            # Combine the small, medium, and large vessels
            combined_vessels = combine_vessel_channels(
                small_vessels_cleaned, medium_vessels_cleaned, large_vessels_cleaned,
                small_weight=1.0, medium_weight=1.0, large_weight=1.0
            )

            # Binarize and clean the combined vessels
            binary_vessels = binarize_combined_vessels(combined_vessels)
            cleaned_vessels = remove_noise(binary_vessels, min_size=150)

            no_gaps_vessels = remove_gaps_in_vessels((cleaned_vessels * 255).astype(np.uint8))
            # Save the final cleaned vessel image
            output_path = os.path.join(output_directory, f'processed_{filename}')
            cv2.imwrite(output_path, no_gaps_vessels)

# Example usage
input_directory = 'C:\\Users\\aliha\\OneDrive - Thammasat University\\Desktop\\OCTA\\Vessel Extraction\\Dataset\\TV_TUH'
output_directory = 'C:\\Users\\aliha\\OneDrive - Thammasat University\\Desktop\\OCTA\\Vessel Extraction\\Dataset\\TV_TUH_processed'
process_and_save_images(input_directory, output_directory)
