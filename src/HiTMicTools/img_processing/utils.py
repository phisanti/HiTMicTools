import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Union, Optional, Dict

def map_predictions_to_labels(
    labeled_image: np.ndarray,
    predictions: Union[List[str], List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    value_map: Optional[Dict[str, int]] = None,
    background_value: int = 0,
) -> np.ndarray:
    """
    Maps prediction values onto a labeled image based on label IDs.

    This function creates a new image where each labeled region is assigned
    a value corresponding to its prediction class. This is useful for
    visualizing classification results directly on the segmented image.

    Args:
        labeled_image: Integer-labeled image where each object has a unique ID.
            Shape can be 2D (single image) or 3D (time series of images).
        predictions: List or array of prediction values (strings or integers)
            corresponding to each label in 'labels'.
        labels: List or array of label IDs that correspond to the predictions.
            Must be the same length as 'predictions'.
        value_map: Optional dictionary mapping string prediction values to integers.
            If None, predictions are assumed to be integers or convertible to integers.
        background_value: Value to assign to background pixels (where labeled_image == 0).
            Defaults to 0.

    Returns:
        np.ndarray: A new image with the same shape as labeled_image, where each
            labeled region is filled with its corresponding prediction value.

    Example:
        >>> # Map cell types onto a labeled image
        >>> cell_types = ["single-cell", "clump", "noise"]
        >>> label_ids = [1, 2, 3]
        >>> type_map = {"single-cell": 1, "clump": 2, "noise": 3}
        >>> type_image = map_predictions_to_labels(labeled_img, cell_types, label_ids, type_map)

        >>> # Map binary classification (e.g., PI positive/negative) onto a labeled image
        >>> pi_status = ["piPOS", "piNEG", "piPOS"]
        >>> label_ids = [1, 2, 3]
        >>> pi_map = {"piPOS": 1, "piNEG": 2}
        >>> pi_image = map_predictions_to_labels(labeled_img, pi_status, label_ids, pi_map)
    """
    # Validate inputs
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    # Convert predictions to integers if a value map is provided
    if value_map is not None:
        pred_values = np.array(
            [value_map.get(pred, background_value) for pred in predictions]
        )
    else:
        # Try to convert predictions to integers directly
        try:
            pred_values = np.array([int(pred) for pred in predictions])
        except (ValueError, TypeError):
            raise ValueError(
                "Predictions must be integers or convertible to integers if no value_map is provided"
            )

    # Create a mapping from label IDs to prediction values
    label_to_pred = {label: pred for label, pred in zip(labels, pred_values)}

    # Define a vectorized function to map labels to predictions
    def map_label_to_pred(x):
        if x == 0:  # Background
            return background_value
        return label_to_pred.get(
            x, background_value
        )  # Default to background_value if label not found

    vectorized_map = np.vectorize(map_label_to_pred)

    # Apply the mapping to create the prediction image
    prediction_image = vectorized_map(labeled_image)

    return prediction_image


def measure_background_intensity(
    img: np.ndarray, mask: np.ndarray, target_channel: int, quantile: float = 0.10
) -> pd.DataFrame:
    """Measure background fluorescence intensity excluding foreground objects.

    Args:
        img (np.ndarray): Image stack [frame, slice, channel, x, y].
        mask (np.ndarray): Binary mask with objects as pixels and background as 0 [frame, slice, x, y].
        target_channel (int): Channel to measure background intensity.

    Returns:
        pd.DataFrame: DataFrame with background intensity (quantile 10) per frame.
    """
    bck_intensities = []
    frames = []
    for frame in range(img.shape[0]):
        # Ensure mask has the same number of dimensions as the image for broadcasting
        frame_mask = mask[frame, 0]
        frame_img = img[frame, 0, target_channel]

        # Apply mask to the image: set object pixels to NaN
        masked_img = np.where(frame_mask == 0, frame_img, np.nan)

        # Calculate the 10th percentile of the background intensity
        bck_intensity = np.nanquantile(masked_img, quantile)
        bck_intensities.append(bck_intensity)
        frames.append(frame)

    # Create a Pandas DataFrame to store the results
    bck_fl_df = pd.DataFrame({"frame": frames, "background": bck_intensities})
    return bck_fl_df


@torch.compile(backend="eager")
def dynamic_resize_roi(image: torch.Tensor, min_size: int) -> torch.Tensor:
    """
    Resize a region of interest (ROI) to a uniform size using PyTorch.
    
    This function resizes the input image to fit within min_size while maintaining
    aspect ratio, then pads it to exactly min_size x min_size.
    
    Args:
        image: Input tensor of shape (Z, H, W) or (H, W)
        min_size: Target size for the output image
        
    Returns:
        torch.Tensor: Resized and padded image of size (min_size, min_size)
    """
    # Check if the image is 3D (Z, H, W)
    is_3d = len(image.shape) == 3
    if is_3d:
        image = image[0]
    
    h, w = image.shape
    
    if h > min_size or w > min_size:
        # Calculate scaling to maintain aspect ratio (only downscale, never upscale)
        scale = min_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Add batch and channel dimensions for interpolation
        image = image.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        image = image.squeeze(0).squeeze(0)  # [1,1,H,W] -> [H,W]
    
    # Pad to target size
    pad_h = max(0, min_size - image.shape[0])
    pad_w = max(0, min_size - image.shape[1])
    
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return image
