import cv2
import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from pystackreg import StackReg
from basicpy import BaSiC
from HiTMicTools.utils import (
    adjust_dimensions,
    stack_indexer,
    round_to_odd,
    unit_converter,
    get_bit_depth,
)


class ImagePreprocessor:
    """
    A class for preprocessing images.

    Args:
        img (np.ndarray): Input image array.
        pixel_size (float, default=1): Pixel size of the image.
        stack_order (str, default='TSCXY'): Order of dimensions in the image stack.
        nchannels (int, default=1): Number of channels in the image.
        metadata (Optional[Dict[str, Union[float, str, int]]], default=None): Image metadata.
    """

    def __init__(
        self,
        img: np.ndarray,
        pixel_size: float = 1,
        stack_order: str = "TSCXY",
        nchannels: int = 1,
        metadata: Optional[Dict[str, Union[float, str, int]]] = None,
    ):
        img = adjust_dimensions(img, stack_order)
        self.img_original = img
        self.img = img
        self.frames_size = img.shape[0]
        self.slices_size = img.shape[1]
        self.channels_size = img.shape[2]

        # Image metadata
        self.bit_depth = get_bit_depth(img)
        if metadata is None:
            self.pixel_size = pixel_size
            self.stack_order = stack_order
            self.nchannels = nchannels
        else:
            self.pixel_size = metadata["pixel_size"]
            self.stack_order = metadata["stack_order"]
            self.nchannels = metadata["nchannels"]

    def align_image(
        self, nchannel: int, nslices: int, crop_image: bool = True, **kwargs
    ) -> None:
        """
        Align the image using a reference channel.

        Args:
            nchannel (int): Reference channel index.
            nslices (int): Reference slice index.
            crop_image (bool, default=True): Whether to crop the black region after alignment.
            **kwargs: Additional keyword arguments for StackReg.

        Returns:
            None
        """
        # Align with reference channel
        reference_channel = self.img[:, nslices, nchannel, :, :]
        sr = StackReg(StackReg.TRANSLATION)
        self.tmats = sr.register_stack(reference_channel, **kwargs)

        # Apply transform to other channels/slices
        index_table = stack_indexer(
            range(self.frames_size), range(self.slices_size), range(self.channels_size)
        )
        index_table_sc = [(s, c) for t, s, c in index_table]
        index_table_sc = set(index_table_sc)

        for c, s in index_table_sc:
            c, s = c - 1, s - 1
            img_stack = self.img[:, c, s, :, :]
            reg_stack = sr.transform_stack(img_stack, tmats=self.tmats)
            self.img[:, c, s, :, :] = reg_stack

        if crop_image:
            min_projection = np.min(reference_channel, axis=0)
            start_h, end_h, start_w, end_w = crop_black_region(min_projection)
            self.img = self.img[:, :, :, start_h:end_h, start_w:end_w]

    def clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        method: str,
        convert_32: bool = True,
        **kwargs,
    ) -> None:
        """
        Clear the image background using the specified method.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            method (str): Background removal method ('divide', 'subtract', or 'basicpy').
            convert_32 (bool, default=True): Whether to convert the image to float32 before processing.
            **kwargs: Additional keyword arguments for the background removal method.

        Returns:
            None
        """
        # Note, in order to collect the image in the self.img, I have to change type before processing
        if convert_32:
            self.img = self.img.astype(np.float32)

        # Assert that nframes, nslices, and nchannels are within valid range
        self.check_size_limit(nframes, self.frames_size, "nframes")
        self.check_size_limit(nslices, self.slices_size, "nslices")
        self.check_size_limit(nchannels, self.channels_size, "nchannels")

        if method == "divide":
            self._cv2_clear_image_background(
                nframes, nslices, nchannels, method="divide", **kwargs
            )
        elif method == "subtract":
            self._cv2_clear_image_background(
                nframes, nslices, nchannels, method="subtract", **kwargs
            )
        elif method == "basicpy":
            self._basicpy_clear_image_background(nframes, nslices, nchannels, **kwargs)
        else:
            raise ValueError(
                f"Invalid method: {method}. Choose either 'divide', 'subtract', or 'basicpy'."
            )

    def _basicpy_clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        **kwargs,
    ) -> None:
        """
        Clear the image background using the basicpy method.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            **kwargs: Additional keyword arguments for BaSiC.

        Returns:
            None
        """
        img_to_transform = self.img[nframes, nslices, nchannels]
        if len(img_to_transform.shape) > 3:
            raise TypeError(
                f"Image to transform with basicpy must be 2D or 3D. Got shape: {img_to_transform.shape}"
            )
        elif len(img_to_transform.shape) == 3:
            is_timelapse = True
        else:
            is_timelapse = False

        basic = BaSiC(**kwargs)
        basic.fit(img_to_transform)
        images_transformed = basic.transform(img_to_transform, timelapse=is_timelapse)
        self.img[nframes, nslices, nchannels] = images_transformed

    def _cv2_clear_image_background(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        method: str,
        **kwargs,
    ) -> None:
        """
        Clear the image background using the OpenCV method.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            method (str): Background removal method ('divide' or 'subtract').
            **kwargs: Additional keyword arguments for clear_background.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)
        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :] = clear_background(
                self.img[t, s, c, :, :], method=method, **kwargs
            )

    def norm_eq_hist(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
    ) -> None:
        """
        Normalize and equalize the histogram of the image.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)

        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :] = norm_eq_hist(self.img[t, s, c, :, :])

    @staticmethod
    def check_size_limit(
        input: Union[int, range, List[int]], size_limit: int, name: str
    ) -> None:
        """
        Check if the input size is within the valid range.

        Args:
            input (Union[int, range, List[int]]): Input size.
            size_limit (int): Maximum allowed size.
            name (str): Name of the input parameter.

        Returns:
            None
        """
        if isinstance(input, int):
            max_value = input
        elif isinstance(input, range):
            max_value = input.stop
        elif isinstance(input, list) and all(isinstance(i, int) for i in input):
            max_value = max(input)
        else:
            raise TypeError(
                f"{name} must be an integer, a range object, or a list of integers"
            )

        assert max_value <= size_limit, f"{name} exceeds image dimensions"


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


if __name__ == "__main__":
    img = np.random.rand(10, 3, 100, 100)

    x = adjust_dimensions(img, "TCXY")
    #    print(x[0, 0, 0, :, :])
    ip = ImagePreprocessor(img, stack_order="TCXY")
    ip.align_image(0, 0, crop_image=True, reference="previous")
    ip.clear_image_background(0, 0, 0, sigma_r=4)

    print("print processed image")
#    print(ip.img[0, 0, 0, :, :])
