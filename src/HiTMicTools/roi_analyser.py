import json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label
from scipy.stats import skew, linregress
from scipy.spatial.distance import pdist
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table
from HiTMicTools.utils import adjust_dimensions, stack_indexer


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
    try:
        contours, _ = cv2.findContours(
            regionmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get perimeter length and convex hull length
        perimeter = cv2.arcLength(contours[0], True)
        hull = cv2.convexHull(contours[0])
        hull_perimeter = cv2.arcLength(hull, True)

        # Calculate the border complexity
        if hull_perimeter != 0:
            border_complexity = perimeter / hull_perimeter
        else:
            border_complexity = 1.0
    except ValueError:
        border_complexity = 0.0

    return border_complexity


## Auxiliary functions
def rod_shape_coef(regionmask, intensity):
    """
    Skeletonize the region mask, perform a fast linear regression, and return the R-squared value.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values (unused in this function).

    Returns:
    float: The R-squared value of the linear regression on the skeletonized region.
    """
    # Get skeletopn coords
    skeleton = skeletonize(regionmask)
    y, x = np.where(skeleton)

    if len(x) < 2:
        return 0.0

    if np.all(x == x[0]):
        return 1.0

    # Calculate the R-squared value
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value**2
    except ValueError:
        r_squared = 0

    return r_squared


def coords_centroid(coords):
    centroid = np.mean(coords, axis=0)
    return pd.Series(centroid, index=["slice", "y", "x"])


def quartiles(regionmask, intensity):
    """
    Calculate the quartiles of the given intensity values within the specified region mask.

    Parameters:
    regionmask (ndarray): A boolean mask indicating the region of interest.
    intensity (ndarray): An array of intensity values.

    Returns:
    ndarray: An array containing the 25th, 50th, and 75th percentiles of the intensity values within the region mask.
    """
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def roi_skewness(regionmask, intensity):
    """
    Calculate the skewness of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.

    Returns:
    float: The skewness of pixel intensities within the ROI.
    """
    roi_intensities = intensity[regionmask]

    try:
        # Check if there are enough unique values in roi_intensities
        unique_values = np.unique(roi_intensities)
        if len(unique_values) < 10:
            return 0

        return skew(roi_intensities, bias=False)
    except Exception:
        return 0


def roi_std_dev(regionmask, intensity):
    """
    Calculate the standard deviation of pixel intensities within a region of interest (ROI).

    Parameters:
    regionmask (numpy.ndarray): A binary mask defining the ROI.
    intensity (numpy.ndarray): The intensity image.

    Returns:
    float: The standard deviation of pixel intensities within the ROI.
    """
    roi_intensities = intensity[regionmask]
    return np.std(roi_intensities)


def laplacian(image):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian.
    """
    image = np.float32(image)

    # Check the data type of the image
    if image.dtype == np.float32:
        ddepth = cv2.CV_32F
    elif image.dtype == np.float64:
        ddepth = cv2.CV_64F
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")

    return cv2.Laplacian(image, ddepth)


def variance_filter(image, kernel_size):
    # Convert the image to float32
    image = np.float32(image)

    # Calculate the mean of the image
    mean = cv2.blur(image, (kernel_size, kernel_size))
    mean_sqr = cv2.blur(np.square(image), (kernel_size, kernel_size))

    # Calculate the variance
    variance = mean_sqr - np.square(mean)

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
    # Ensure regionmask is an 8-bit, single-channel image
    regionmask = regionmask.astype(np.uint8)

    # Dilate the regionmask
    dilated_regionmask = cv2.dilate(regionmask, structure, iterations=iterations)

    # Get the intensities within the dilated ROI
    roi_intensities = intensity[dilated_regionmask > 0]
    std_px = np.std(roi_intensities)
    mean_px = np.mean(roi_intensities)
    min_px = np.min(roi_intensities)
    max_px = np.max(roi_intensities)
    pixel_area = np.sum(dilated_regionmask > 0)

    return (std_px, mean_px, min_px, max_px, pixel_area)


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

    def get_labels(self):
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

        self.total_rois = num_rois
        self.labeled_mask = labeled_mask

    def get_roi_measurements(
        self,
        target_channel=0,
        target_slice=0,
        properties=["label", "centroid", "mean_intensity"],
        extra_properties=None,
        asses_focus=True,
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
            roi_properties_df = pd.DataFrame(roi_properties)

            if asses_focus:
                var_colnames = {
                    "mean_intensity": "mean_0",
                    "dilated_measures_0": "std",
                    "dilated_measures_1": "mean",
                    "dilated_measures_2": "min",
                    "dilated_measures_3": "max",
                    "dilated_measures_4": "area",
                }
                var_img = variance_filter(img_frame, 10)
                lap_im = laplacian(img_frame)

                roi_var = regionprops_table(
                    labeled_mask_frame,
                    intensity_image=var_img,
                    properties=["mean_intensity"],
                    extra_properties=(dilated_measures,),
                    separator="_",
                )
                roi_lap = regionprops_table(
                    labeled_mask_frame,
                    intensity_image=lap_im,
                    properties=["mean_intensity"],
                    extra_properties=(dilated_measures,),
                    separator="_",
                )
                roi_var = pd.DataFrame(roi_var)
                roi_lap = pd.DataFrame(roi_lap)
                roi_lap.columns = [
                    "lap_" + var_colnames.get(col, col) for col in roi_lap.columns
                ]
                roi_var.columns = [
                    "var_" + var_colnames.get(col, col) for col in roi_var.columns
                ]

                roi_properties_df = pd.concat(
                    [roi_properties_df, roi_var, roi_lap], axis=1
                )

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
