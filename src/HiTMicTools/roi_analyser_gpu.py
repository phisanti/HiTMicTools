import json
import numpy as np
import cupy as cp
import pandas as pd
import cv2

import itertools
from typing import Union, List


from cupyx.scipy import ndimage
from cupyx.scipy import stats

from scipy.ndimage import label
from scipy.stats import skew, linregress
from scipy.spatial.distance import pdist
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table

def border_complexity(regionmask, intensity):
    """
    Calculate the border complexity of a region by comparing the perimeter of the region
    to the perimeter of its convex hull.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values (unused in this function).

    Returns:
    float: The border complexity value, defined as the ratio of the region's perimeter
           to the perimeter of its convex hull.
    """

    # TODO: Calculate the perimeter of the region using cv2.contourPerimeter


    return border_complexity

def rod_shape_coef(regionmask, intensity):
    """
    Skeletonize the region mask, perform a fast linear regression, and return the R-squared value.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values (unused in this function).

    Returns:
    float: The R-squared value of the linear regression on the skeletonized region.
    """
    # Get skeleton coords
    skeleton = ndimage.morphology.skeletonize(regionmask)
    y, x = cp.where(skeleton)

    if len(x) < 2:
        return 0.0

    if cp.all(x == x[0]):
        return 1.0

    # Calculate the R-squared value
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.get(), y.get())
        r_squared = r_value**2
    except ValueError:
        r_squared = 0

    return r_squared

def coords_centroid(coords):
    centroid = cp.mean(coords, axis=0)
    return pd.Series(centroid.get(), index=["slice", "y", "x"])


def quartiles(regionmask, intensity):
    """
    Calculate the quartiles of the given intensity values within the specified region mask.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values.

    Returns:
    ndarray: An array containing the 25th, 50th, and 75th percentiles of the intensity values within the region mask.
    """
    return cp.percentile(intensity[regionmask], q=(25, 50, 75))


def roi_skewness(regionmask, intensity):
    """
    Calculate the skewness of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (cupy.ndarray): A binary mask defining the ROI.
    intensity (cupy.ndarray): The intensity image.

    Returns:
    float: The skewness of pixel intensities within the ROI.
    """
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
    """
    Calculate the standard deviation of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (cupy.ndarray): A binary mask defining the ROI.
    intensity (cupy.ndarray): The intensity image.

    Returns:
    float: The standard deviation of pixel intensities within the ROI.
    """
    roi_intensities = intensity[regionmask]
    return float(cp.std(roi_intensities))


def laplacian(image):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian.
    """

    # TODO:

    return cv2.Laplacian(image, ddepth)


def variance_filter(image, kernel_size):
    # Convert the image to float32

    # TODO: 
    return variance

def dilated_measures(regionmask, intensity, structure=np.ones((5, 5)), iterations=1):
    """
    Calculate the standard deviation of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.
    structure (numpy.ndarray): The structuring element used for dilation.
    iterations (int): The number of times dilation is applied.

    Returns:
    float: The standard deviation of pixel intensities within the ROI.
    """
    #TODO:
    return (std_px, mean_px, min_px, max_px, pixel_area)


import cupy as cp

def adjust_dimensions_ingpu(img: cp.ndarray, dim_order: str) -> cp.ndarray:
    """
    Adjust the dimensions of an image array to match the target order 'TSCXY'.

    Args:
        img (cp.ndarray): Input image array.
        dim_order (str): Current dimension order of the image. Should be a permutation or subset of 'TSCXY'.
                         T=Time, S=Slice, C=Channel, X=Width, Y=Height.

    Returns:
        cp.ndarray: Image array with adjusted dimensions.
    """

    target_order = "TSCXY"
    assert set(dim_order).issubset(
        set(target_order)
    ), "Invalid dimension order. Allowed dimensions: 'TSCXY'"

    missing_dims = set(target_order) - set(dim_order)

    # Add missing dimensions
    for dim in missing_dims:
        index = target_order.index(dim)
        img = cp.expand_dims(img, axis=index)
        dim_order = dim_order[:index] + dim + dim_order[index:]

    # Reorder dimensions
    order = [dim_order.index(dim) for dim in target_order]
    img = cp.transpose(img, order)

    return img


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
        image = stack_indexer_ingpu(image, stack_order[0])
        probability_map = stack_indexer_ingpu(probability_map, stack_order[1])

        self.img = image
        self.proba_map = probability_map
        self.stack_order = stack_order

        pass

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