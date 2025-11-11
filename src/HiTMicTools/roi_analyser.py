# Standard library imports
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.measure import regionprops_table

# Local imports
from HiTMicTools.img_processing.array_ops import adjust_dimensions

# Type hints
from numpy.typing import NDArray
from pandas import DataFrame, Series


def coords_centroid(coords):
    centroid = np.mean(coords, axis=0)
    return pd.Series(centroid, index=["slice", "y", "x"])


def convert_to_list_and_dump(row):
    return json.dumps(row.tolist())


class RoiAnalyser:
    def __init__(self, image, probability_map, stack_order=("TSCXY", "TXY")):
        image = adjust_dimensions(image, stack_order[0])
        probability_map = adjust_dimensions(probability_map, stack_order[1])

        self.img = image
        self.proba_map = probability_map
        self.stack_order = stack_order

        pass

    @classmethod
    def from_labeled_mask(
        cls,
        image: np.ndarray,
        labeled_mask: np.ndarray,
        stack_order: Tuple[str, str] = ("TSCXY", "TYX"),
    ) -> "RoiAnalyser":
        """
        Create RoiAnalyser directly from a pre-labeled mask, skipping probability-based segmentation.

        This constructor is useful when you have instance segmentation outputs (e.g., from RF-DETR-Segm)
        that already provide labeled instances, bypassing the need for probability maps and
        connected components analysis.

        Args:
            image: The original microscopy image with shape matching stack_order[0].
            labeled_mask: Pre-labeled instance mask where each unique positive integer
                represents a distinct ROI. Shape should match stack_order[1].
            stack_order: Tuple of (image_order, mask_order) dimension specifications.
                Defaults to ("TSCXY", "TYX") for time-series images with pre-labeled masks.

        Returns:
            RoiAnalyser instance ready for measurements, with labeled_mask already populated.

        Example:
            >>> # From RF-DETR-Segm output
            >>> labeled_mask, _, _, _ = sc_segmenter.predict(image)
            >>> analyser = RoiAnalyser.from_labeled_mask(image, labeled_mask)
            >>> measurements = analyser.get_roi_measurements(target_channel=1)
        """
        instance = cls.__new__(cls)

        # Adjust dimensions to expected format
        instance.img = adjust_dimensions(image, stack_order[0])
        adjusted_mask = adjust_dimensions(labeled_mask, stack_order[1])

        # Set attributes
        instance.labeled_mask = adjusted_mask
        instance.stack_order = stack_order
        instance.proba_map = None  # No probability map in this workflow

        # Derive binary mask from labeled mask
        instance.binary_mask = adjusted_mask > 0

        # Calculate total number of ROIs across all frames
        instance.total_rois = int(np.max(adjusted_mask))

        return instance

    def create_binary_mask(self, threshold=0.5):
        """
        Create a binary mask from an image stack of probabilities.

        Args:
            image_stack (np.ndarray): A 5D numpy array with shape (frames, slices, channels, height, width) containing probabilities.
            threshold (float): The threshold value to use for binarization (default: 0.5).

        Returns:
            np.ndarray: A 5D numpy array with the same shape as the input, where values above the threshold are set to 1, and values below or equal to the threshold are set to 0.
        """

        # Convert probabilities to binary values
        self.binary_mask = self.proba_map > threshold

    def clean_binmask(self, min_pixel_size=20):
        """
        Clean the binary mask by removing small ROIs.
        Args:
            min_pixel_size (int): Minimum ROI size in pixels.

        Returns:
            cleaned_mask (np.ndarray): Cleaned binary mask.
        """
        labeled, num_features = self.get_labels(return_value=True)
        sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (label 0)
        mask_sizes = sizes >= min_pixel_size
        label_map = np.zeros(num_features + 1, dtype=int)
        label_map[1:][mask_sizes] = np.arange(1, np.sum(mask_sizes) + 1)
        cleaned_labeled = label_map[labeled]
        cleaned_mask = cleaned_labeled > 0
        self.binary_mask = cleaned_mask

    def get_labels(self, return_value=False):
        """
        Get the labeled mask for the binary mask.

        Returns:
            None
        """
        labeled_mask = np.empty_like(self.binary_mask, dtype=int)
        num_rois = 0
        max_label = 0

        for i in range(self.binary_mask.shape[0]):
            labeled_frame, num = label(self.binary_mask[i])
            labeled_mask[i] = np.where(
                labeled_frame != 0, labeled_frame + max_label, labeled_frame
            )
            max_label += num
            num_rois += num

        if return_value:
            return labeled_mask, num_rois
        else:
            self.total_rois = num_rois
            self.labeled_mask = labeled_mask

    def get_roi_measurements(
        self,
        target_channel=0,
        target_slice=0,
        properties=["label", "centroid", "mean_intensity"],
        extra_properties=None,
    ):
        """
        Get measurements for each ROI in the labeled mask for a specific channel and all frames.

        Args:
            img (numpy.ndarray): The original image.
            labeled_mask (numpy.ndarray): The labeled mask containing the ROIs.
            properties (list, optional): A list of properties to measure for each ROI.
                Defaults to ['mean_intensity', 'centroid'].

        Returns:
            list: A list of dictionaries, where each dictionary contains the measurements
                for a single ROI.
        """

        assert self.labeled_mask is not None, (
            "Run get_labels() first to generate labeled mask"
        )

        img = self.img[:, target_slice, target_channel, :, :]
        labeled_mask = self.labeled_mask[:, target_slice, 0, :, :]

        all_roi_properties = []

        for frame in range(img.shape[0]):
            img_frame = img[frame]
            labeled_mask_frame = labeled_mask[frame]

            roi_properties = regionprops_table(
                labeled_mask_frame,
                intensity_image=img_frame,
                properties=properties,
                separator="_",
                extra_properties=extra_properties,
            )
            roi_properties_df = pd.DataFrame(roi_properties)
            roi_properties_df["frame"] = frame
            roi_properties_df["slice"] = target_channel
            roi_properties_df["channel"] = target_slice

            all_roi_properties.append(roi_properties_df)

        all_roi_properties_df = pd.concat(all_roi_properties, ignore_index=True)

        if "coords" in all_roi_properties_df.columns:
            all_roi_properties_df["coords"] = all_roi_properties_df["coords"].apply(
                convert_to_list_and_dump
            )

        # rearrange the columns
        required_cols = ["label", "frame", "slice", "channel"]
        other_cols = [
            col for col in all_roi_properties_df.columns if col not in required_cols
        ]

        cols = required_cols + other_cols
        all_roi_properties_df = all_roi_properties_df[cols]

        return all_roi_properties_df
