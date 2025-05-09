from typing import List, Union, Optional, Dict
import numpy as np


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
