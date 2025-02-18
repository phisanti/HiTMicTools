import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from typing import Any
from .image_scaler import ImageScaler
from .base_model import BaseModel
from monai.inferers import SlidingWindowInferer
from HiTMicTools.utils import get_device


class FocusRestorer(BaseModel):
    """
    A class representing a segmentation model.

    Attributes:
        model (torch.nn.Module): The segmentation model.
        device (torch.device): The device on which the model is loaded.
    """

    def __init__(
        self,
        model_path: str,
        model_graph: torch.nn.Module,
        patch_size: int,
        overlap_ratio: float,
        scale_method='range01',
        half_precision: bool = False,
    scaler_args: dict = {}

    ) -> None:
        """
        Initializes a Segmentator object.

        Args:
            model_path (str): The path to the saved model weights.
            model_graph (torch.nn.Module): The model architecture.
            patch_size (int): The size of the patches for sliding window inference.
            overlap_ratio (float): The overlap ratio between patches for sliding window inference.
            scale_method (range01, eq-centered or none): The method for image scaling before inference.
            half_precision (bool, optional): Whether to use half-precision (FP16) for inference. Defaults to False.
        """

        assert (
            isinstance(patch_size, int) and patch_size > 0
        ), "patch_size must be a positive integer"
        assert (
            isinstance(overlap_ratio, float) and 0 <= overlap_ratio < 1
        ), "overlap_ratio must be a float between 0 and 1"

        assert scale_method in ['range01', 'zscore', 'combined', 'fixed_range'], "Invalid scale method"
        # Validate scaler arguments based on method
        if scale_method == 'fixed_range':
            assert 'bit_depth' in scaler_args, "bit_depth required for fixed_range scaling"
        elif scale_method in ['range01', 'combined']:
            if 'pmin' in scaler_args and 'pmax' in scaler_args:
                assert 0 <= scaler_args['pmin'] < scaler_args['pmax'] <= 100, "pmin/pmax must be 0 <= pmin < pmax <= 100"



        self.device = get_device()
        self.model = self.get_model(model_path, self.device, model_graph=model_graph)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.model.eval()
        self.half_precision = half_precision
        if scale_method is not None:
            scaler_args['scale_method'] = scale_method
            scaler_args['device'] = self.device
            self.scaler = ImageScaler(**scaler_args)

        if self.half_precision:
            self.model.half()
        else:
            self.model.float()

    def predict(self, image: np.ndarray, is_3D: bool=False, return_tensor: bool=False, rescale: bool=True, **kwargs: Any) -> np.ndarray:
        """
        Predicts the segmentation mask for the given image. It can handle 2D images or a stack of 2D images.
        This method connects the model with the monai SlidingWindowInferer to perform inference on patches of the input image.
        It also takes care the normalisation and the conversion to tensor.

        Args:
            image (numpy.ndarray): The input image or image stack.
            is_3D (bool): For the analysis of 3D images where the [D, H, W]
            return_tensor (bool): Return the output as a tensor.
            rescale (bool): Rescale the output to the original range.
            **kwargs: Additional keyword arguments to pass to the SlidingWindowInferer.

        Returns:
            numpy.ndarray: The segmentation mask or stack of segmentation masks.
        """
        if torch.is_tensor(image):
            img_tensor = image.to(self.device)
        else:
            img_tensor = torch.from_numpy(image).to(self.device)

        # Prepare image
        image, added_dim_index  = self.ensure_4d(image, is_3D)

        # add scaling methods here
        if self.scale_method is not None:
            img_tensor = self.scaler.scale_image(img_tensor)
    
        if self.half_precision:
            img_tensor = img_tensor.half()  # Convert input to half-precision
        else:
            img_tensor = img_tensor.float()

        # Create SlidingWindowInferer
        inferer = SlidingWindowInferer(
            roi_size=self.patch_size, overlap=self.overlap_ratio, **kwargs
        )

        with torch.no_grad():
            output = inferer(img_tensor, self.model)
            if rescale and self.scale_method is not None:
                output = self.scaler.rescale_image(output, self.scale_method)

        # Handle output format
        del img_tensor
        if added_dim_index is not None:
            output = output.squeeze(added_dim_index)

        if return_tensor:
            return output
        
        output_np = output.cpu().numpy()
        del output
        self.cleanup()

        return output_np