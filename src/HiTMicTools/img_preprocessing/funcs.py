import cv2
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from pystackreg import StackReg
from HiTMicTools.utils import (
    round_to_odd,
    unit_converter,
)


def detect_and_fix_well(
    image: np.ndarray,
    darkness_threshold_factor: float = 0.4,
    border_sample_interval: int = 100
) -> Tuple[np.ndarray, bool]:
    """
    Detects and fixes dark well borders from the well plate in microscopy images. The algorithm works as follows:
     1- Check for dark borders using sampled border pixels. (return image if none)
     2- If dark borders are detected, it creates a mask of dark pixels and identifies connected components
     3- Checks which components touch the image borders. 
     4- Finally, it replaces the dark border pixels with the mean pixel value of the non-border regions.
    
    Args:
        image: Input grayscale image
        darkness_threshold_factor: Factor for darkness threshold calculation
        border_sample_interval: Sampling interval for border pixel examination
        
    Returns:
        Tuple[np.ndarray, bool]: A tuple containing the corrected image and a boolean indicating if a border was detected.
    """

    # Quick border check using sample points with early exit for no dark border
    num_labels = 0
    image_mean = np.mean(image)
    border_pixels = np.concatenate([
        image[0, ::50], image[-1, ::50], 
        image[::50, 0], image[::50, -1]
    ])

    if np.min(border_pixels) > image_mean * darkness_threshold_factor:
        return image, False
        
    # Create dark pixel mask
    dark_mask = image < (image_mean * darkness_threshold_factor)
    if not np.any(dark_mask):
        return image, False 
        
    # Find connected components in dark regions
    num_labels, labels = cv2.connectedComponents(dark_mask.astype(np.uint8))
    if num_labels <= 1:
        return image, False 
        
    # Efficiently sample border pixels to detect border components
    border_components = set()
    
    # Check top/bottom borders
    top_samples = labels[0, ::border_sample_interval]
    bottom_samples = labels[-1, ::border_sample_interval]
    border_components.update(top_samples[top_samples > 0])
    border_components.update(bottom_samples[bottom_samples > 0])
    
    # Check left/right borders
    left_samples = labels[::border_sample_interval, 0]
    right_samples = labels[::border_sample_interval, -1]
    border_components.update(left_samples[left_samples > 0])
    border_components.update(right_samples[right_samples > 0])
    
    if not border_components:
        return image, False 
        
    # Create mask of border components and fix image
    border_mask = np.zeros_like(dark_mask)
    for label in border_components:
        border_mask |= (labels == label)
        
    # Fix borders using non-border mean
    fixed_image = np.copy(image)
    non_border_mask = ~border_mask
    non_border_mean = np.mean(image[non_border_mask])
    fixed_image[border_mask] = non_border_mean
    
    return fixed_image, True

def clear_background(
    img,
    sigma_r,
    unit="pixel",
    method="divide",
    pixel_size=1,
    convert_32=True,
    clip_negative=True,
):
    # Input checks
    if img.ndim != 2:
        raise ValueError("Input image must be 2D")
    if convert_32:
        img = img.astype(np.float32)

    if unit == "pixel":
        pass
    else:
        sigma_r = unit_converter(sigma_r, pixel_size, to_unit="pixel")
        sigma_r = int(sigma_r)

    # Gaussian blur
    sigma_r = round_to_odd(sigma_r)
    gaussian_blur = cv2.GaussianBlur(img, (sigma_r, sigma_r), 0)

    # Background remove
    if method == "subtract":
        background_removed = cv2.subtract(img, gaussian_blur)
    elif method == "divide":
        background_removed = cv2.divide(img, gaussian_blur)
    else:
        raise ValueError("Invalid method. Choose either 'subtract' or 'divide'")

    if clip_negative:
        background_removed = np.clip(background_removed, 0, None)

    return background_removed


def convert_to_uint8(image):
    # Normalize the image to the range 0-1
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Scale the normalized image to the range 0-255
    scaled_image = normalized_image * 255

    # Convert the scaled image to uint8
    uint8_image = scaled_image.astype(np.uint8)

    return uint8_image


def norm_eq_hist(img):
    img = convert_to_uint8(img)
    equalized = cv2.equalizeHist(img.astype(np.uint8))
    equalized = equalized.astype(np.float32)
    equalized = (equalized - equalized.mean()) / equalized.std()

    return equalized


# TODO: VECTORISE FUNCTION. IT CAN BE DONE AS SHOWN COMMENTED BELOW THE CURRENT FUNCTION
def crop_black_region(img):
    h, w = img.shape
    while True:
        start_h = (img.shape[0] - h) // 2
        start_w = (img.shape[1] - w) // 2
        roi = img[start_h : start_h + h, start_w : start_w + w]
        if np.any(roi == 0.0):
            h -= 1
            w -= 1
        else:
            break

    end_h = start_h + h
    end_w = start_w + w
    return start_h, end_h, start_w, end_w
