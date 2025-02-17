
def calculate_dice_score(coords1, coords2):
    """
    Calculate the Dice Score between two sets of pixel coordinates.

    Args:
        coords1 (list of tuples): List of (x, y) pixel coordinates for the first image.
        coords2 (list of tuples): List of (x, y) pixel coordinates for the second image.

    Returns:
        float: The Dice Score between the two images.
    """
    set1 = set(coords1)
    set2 = set(coords2)
    
    intersection = set1 & set2
    intersection_size = len(intersection)
    set1_size = len(set1)
    set2_size = len(set2)
    
    # Calculate Dice Score
    dice_score = (2 * intersection_size) / (set1_size + set2_size)
    return dice_score