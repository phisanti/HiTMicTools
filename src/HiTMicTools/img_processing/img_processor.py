import cv2
import numpy as np
from typing import Union, List, Optional, Dict
from pystackreg import StackReg
from basicpy import BaSiC
from .img_ops import (
    clear_background,
    norm_eq_hist,
    crop_black_region,
    detect_and_fix_well,
)
from .array_ops import (
    adjust_dimensions,
    stack_indexer,
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
        self,
        ref_channel: int,
        ref_slice: int,
        compres_align: float = 0,
        normalise_image: bool = True,
        crop_image: bool = True,
        **kwargs,
    ) -> None:
        """
        Align the image using a reference channel.

        Args:
            ref_channel (int): Reference channel index.
            ref_slice (int): Reference slice index.
            compres_align (float, default=0): Compression ratio to crop image for alignment. Speed up
            alignment for very large images and long stacks.
            crop_image (bool, default=True): Whether to crop the black region after alignment.
            **kwargs: Additional keyword arguments for StackReg.

        Returns:
            None
        """
        # check that compress_align is between 0-1
        assert compres_align >= 0 and compres_align <= 1, (
            "compress_align must be between 0 and 1"
        )

        # Align with reference channel
        reference_channel = self.img[:, ref_slice, ref_channel, :, :]
        
        if normalise_image:
            reference_channel = reference_channel / np.mean(reference_channel, axis=(1, 2), keepdims=True)

        if compres_align > 0:
            b, h, w = reference_channel.shape
            h_start = int(h * compres_align / 2)
            w_start = int(w * compres_align / 2)
            h_end = h - h_start
            w_end = w - w_start
            reference_channel = reference_channel[:, h_start:h_end, w_start:w_end]

        self.sr = StackReg(StackReg.TRANSLATION)
        self.tmats = self.sr.register_stack(reference_channel, **kwargs)

        # Apply transform to other channels/slices
        index_table = stack_indexer(0, range(self.slices_size), range(self.channels_size))

        for _t, s, c in index_table:
            img_stack = self.img[:, s, c, :, :]
            registered_stack = self.sr.transform_stack(img_stack, tmats=self.tmats)
            self.img[:, s, c, :, :] = registered_stack

        if crop_image:
            # If compressed, resize to the original size
            if compres_align > 0:
                reg_stack = self.img[:, ref_channel, ref_slice]
                original_h, original_w = self.img.shape[-2:]
                min_projection_compressed = np.min(reg_stack, axis=0)
                min_projection = cv2.resize(
                    min_projection_compressed, 
                    (original_w, original_h), 
                    interpolation=cv2.INTER_LINEAR
                )
                start_h, end_h, start_w, end_w = crop_black_region(min_projection)
            else:
                min_projection = np.min(reference_channel, axis=0)
                start_h, end_h, start_w, end_w = crop_black_region(min_projection)

            self.img = self.img[:, :, :, start_h:end_h, start_w:end_w]

    def align_from_matrix(self, img: np.ndarray) -> np.ndarray:
        """
        Align a new image using a transformation matrix from the source image.
        Note, the reference image should be the same as the one used to create the transformation matrix.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            np.ndarray: Aligned image array.
        """
        # Check if the input image has the correct dimensions
        if (
            img.ndim != 3
            or img.shape[0] != self.frames_size
            or img.shape[1:] != self.img.shape[-2:]
        ):
            raise ValueError(
                f"Input image must be 3D (frames, x, y) with shape ({self.frames_size}, {self.img.shape[-2]}, {self.img.shape[-1]}). "
                f"Got shape: {img.shape}"
            )

        if self.tmats is None:
            raise ValueError(
                "Transformation matrix is not available. Please align the reference image first."
            )

        return self.sr.transform_stack(img, tmats=self.tmats)


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

    def detect_fix_well(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
        **kwargs,
    ) -> None:
        """
        Detect and fix well borders in the image. It update the loaded image and save the border detection info for logging purposes.

        Args:
            nframes (Union[int, range, List[int]]): Frame indices to process.
            nslices (Union[int, range, List[int]]): Slice indices to process.
            nchannels (Union[int, range, List[int]]): Channel indices to process.
            **kwargs: Additional keyword arguments for detect_and_fix_well.

        Returns:
            None
        """
        index_table = stack_indexer(nframes, nslices, nchannels)
        has_border_array = np.zeros(
            (self.frames_size, self.slices_size, self.channels_size), dtype=bool
        )

        for index in index_table:
            t, s, c = index
            self.img[t, s, c, :, :], has_border = detect_and_fix_well(
                self.img[t, s, c, :, :], **kwargs
            )
            has_border_array[t, s, c] = has_border
        self.borders = has_border_array

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

    def scale_channel(
        self,
        nframes: Union[int, range, List[int]],
        nslices: Union[int, range, List[int]],
        nchannels: Union[int, range, List[int]],
    ) -> None:
        """
        Scale the image by (x - mean) / sd for each slice of the target channel.

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
            slice_data = self.img[t, s, c, :, :]
            mean = np.mean(slice_data)
            std = np.std(slice_data)
            self.img[t, s, c, :, :] = (slice_data - mean) / std

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
