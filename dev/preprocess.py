import cv2

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Parameters:
    - image: Input image in RGB format.
    - clip_limit: Threshold for contrast limiting.
    - tile_grid_size: Size of grid for histogram equalization.
    
    Returns:
    - CLAHE applied image.
    """
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v = clahe.apply(v)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image