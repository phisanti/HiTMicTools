# Standard library imports
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cupy as cp
import pandas as pd
from cupyx.scipy import stats
from cupyx.scipy.ndimage import label
from cucim.skimage.measure import regionprops_table
import cudf

# Local imports
from HiTMicTools.utils import adjust_dimensions, stack_indexer

# Type hints
from numpy.typing import NDArray
from pandas import DataFrame, Series

def roi_skewness(regionmask, intensity):
    """Cupy version for the ROI standard deviation as defined in analysis_tools.utils"""
    roi_intensities = intensity[regionmask]

    try:
        # Check if there are enough unique values in roi_intensities
        unique_values = cp.unique(roi_intensities)
        if len(unique_values) < 10:
            return 0

        return float(stats.skew(roi_intensities, bias=False))
    except Exception:
        return 0

def roi_std_dev(regionmask, intensity):
    """Cupy version for the ROI standard deviation as defined in analysis_tools.utils"""
    roi_intensities = intensity[regionmask]
    return float(cp.std(roi_intensities))

def coords_centroid(coords):
    centroid = cp.mean(coords, axis=0)
    return pd.Series(centroid, index=["slice", "y", "x"])


def convert_to_list_and_dump(row):
    return json.dumps(row.tolist())

def stack_indexer_ingpu(
    nframes: Union[int, List[int], range] = [0],
    nslices: Union[int, List[int], range] = [0],
    nchannels: Union[int, List[int], range] = [0],
) -> cp.ndarray:
    """
    Generate an index table for accessing specific frames, slices, and channels in an image stack.
    This aims to simplify the process of iterating over different combinations of frame, slice,
    and channel indices with for loops.

    Args:
        nframes (Union[int, List[int], range], optional): Frame indices. Defaults to [0].
        nslices (Union[int, List[int], range], optional): Slice indices. Defaults to [0].
        nchannels (Union[int, List[int], range], optional): Channel indices. Defaults to [0].

    Returns:
        cp.ndarray: Index table with shape (n_combinations, 3), where each row represents a combination
                    of frame, slice, and channel indices.

    Raises:
        ValueError: If any dimension contains negative integers.
        TypeError: If any dimension is not an integer, list of integers, or range object.
    """
    dimensions = []
    for dimension in [nframes, nslices, nchannels]:
        if isinstance(dimension, int):
            if dimension < 0:
                raise ValueError("Dimensions must be positive integers or lists.")
            dimensions.append([dimension])
        elif isinstance(dimension, (list, range)):
            if not all(isinstance(i, int) and i >= 0 for i in dimension):
                raise ValueError(
                    "All elements in the list dimensions must be positive integers."
                )
            dimensions.append(dimension)
        else:
            raise TypeError(
                "All dimensions must be either positive integers or lists of positive integers."
            )

    combinations = list(itertools.product(*dimensions))
    index_table = cp.array(combinations)
    return index_table

class RoiAnalyser:
    def __init__(self, image, probability_map, stack_order=("TSCXY", "TXY")):
        image = adjust_dimensions(image, stack_order[0])
        probability_map = adjust_dimensions(probability_map, stack_order[1])

        self.img = cp.asarray(image)
        self.proba_map = cp.asarray(probability_map)
        self.stack_order = stack_order


    def create_binary_mask(self, threshold=0.5):
        """
        Create a binary mask from an image stack of probabilities.

        Args:
            image_stack (cp.ndarray): A 5D cupy array with shape (frames, slices, channels, height, width) containing probabilities.
            threshold (float): The threshold value to use for binarization (default: 0.5).

        Returns:
            cupy.ndarray: A 5D numpy array with the same shape as the input, where values above the threshold are set to 1, and values below or equal to the threshold are set to 0.
        """

        # Convert probabilities to binary values
        self.binary_mask = self.proba_map > threshold

    def clean_binmask(self, min_pixel_size=20):
        """
        Clean the binary mask by removing small ROIs.
        Args:
            min_pixel_size (int): Minimum ROI size in pixels.

        Returns:
            cleaned_mask (cp.ndarray): Cleaned binary mask.
        """
        labeled, num_features = self.get_labels(return_value=True)
        sizes = cp.bincount(labeled.ravel())[1:]
        mask_sizes = sizes >= min_pixel_size
        label_map = cp.zeros(num_features + 1, dtype=int)
        label_map[1:][mask_sizes] = cp.arange(1, cp.sum(mask_sizes) + 1)
        cleaned_labeled = label_map[labeled]
        cleaned_mask = cleaned_labeled > 0
        self.binary_mask = cleaned_mask
        
    def get_labels(self, return_value=False):
        """
        Get the labeled mask for the binary mask.

        Returns:
            None
        """
        labeled_mask = cp.empty_like(self.binary_mask, dtype=int)
        num_rois = 0
        max_label = 0

        for i in range(self.binary_mask.shape[0]):
            labeled_frame, num = label(self.binary_mask[i])
            labeled_mask[i] = cp.where(
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
            img (cupy.ndarray): The original image.
            labeled_mask (cupy.ndarray): The labeled mask containing the ROIs.
            properties (list, optional): A list of properties to measure for each ROI.
                Defaults to ['mean_intensity', 'centroid'].

        Returns:
            list: A list of dictionaries, where each dictionary contains the measurements
                for a single ROI.
        """

        assert (
            self.labeled_mask is not None
        ), "Run get_labels() first to generate labeled mask"

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
            roi_properties_df = cudf.DataFrame(roi_properties)
            roi_properties_df["frame"] = frame
            roi_properties_df["slice"] = target_channel
            roi_properties_df["channel"] = target_slice

            all_roi_properties.append(roi_properties_df)

        all_roi_properties_cudf = cudf.concat(all_roi_properties, ignore_index=True)

        if "coords" in all_roi_properties_cudf.columns:
            all_roi_properties_cudf["coords"] = all_roi_properties_cudf["coords"].apply(
                convert_to_list_and_dump
            )

        # rearrange the columns
        required_cols = ["label", "frame", "slice", "channel"]
        other_cols = [
            col for col in all_roi_properties_cudf.columns if col not in required_cols
        ]

        cols = required_cols + other_cols
        all_roi_properties_cudf = all_roi_properties_cudf[cols]

        # Convert to pandas DataFrame at the very end
        all_roi_properties_df = all_roi_properties_cudf.to_pandas()

        return all_roi_properties_df