import torch
import numpy as np
from typing import Any
from monai.inferers import SlidingWindowInferer
from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.model_components.image_scaler import ImageScaler
from HiTMicTools.utils import get_device

class Segmentator(BaseModel):
    """
    A class for performing image segmentation using a pre-trained model.

    The key points of this class is the pre-processing of the input images if required and the 
    SlidingWindow inference in order to predict in batchest to not saturate the GPU RAM.

    Attributes:
        model (torch.nn.Module): The loaded segmentation model.
        device (torch.device): The device on which the model is loaded.
        patch_size (int): The size of the patches used for sliding window inference.
        overlap_ratio (float): The overlap ratio between patches during sliding window inference.
        scale_method (str): The method used for image scaling before inference.
        half_precision (bool): Whether to use half-precision (FP16) for inference.

    Methods:
        __init__(model_path, model_graph, patch_size, overlap_ratio, scale_method, half_precision):
            Initializes the Segmentator with the specified parameters.
        predict(image, is_3D, sigmoid, **kwargs):
            Performs segmentation inference on the input image.
        normalize_percentile_batch(images, pmin, pmax, clip, dtype):
            Applies percentile-based normalization to a batch of images.
        eq_scale(img):
            Applies histogram equalization to the input image.
        to8bit(image):
            Converts the input image to 8-bit format.
    """

    def __init__(
        self,
        model_path: str,
        model_graph: torch.nn.Module,
        patch_size: int,
        overlap_ratio: float,
        scale_method: str = "range01",
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
            scale_method (range01 or eq-centered): The method for image scaling before inference.
            half_precision (bool, optional): Whether to use half-precision (FP16) for inference. Defaults to False.
        """

        assert (
            isinstance(patch_size, int) and patch_size > 0
        ), "patch_size must be a positive integer"
        assert (
            isinstance(overlap_ratio, float) and 0 <= overlap_ratio < 1
        ), "overlap_ratio must be a float between 0 and 1"
        assert scale_method in ['range01', 'zscore', 'combined', 'fixed_range', 'none'], "Invalid scale method"

        self.device = get_device()
        self.model = self.get_model(model_path, model_graph=model_graph, device=self.device)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.scale_method = scale_method
        self.model.eval()
        self.half_precision = half_precision
        if self.half_precision:
            self.model.half()
        else:
            self.model.float()
        if scale_method != 'none':
            scaler_args['scale_method'] = scale_method
            scaler_args['device'] = self.device
            self.scaler = ImageScaler(**scaler_args)

        if self.half_precision:
            self.model.half()
        else:
            self.model.float()
    def predict(
        self,
        image: np.ndarray,
        is_3D: bool = False,
        batch_size: int = None,
        sigmoid: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Predicts the segmentation mask for the given image. It can handle 2D images or a stack of 2D images.
        This method connects the model with the monai SlidingWindowInferer to perform inference on patches of the input image.
        It also takes care the normalisation and the conversion to tensor.

        Args:
            image (numpy.ndarray): The input image or image stack.
            is_3D (bool): For the analysis of 3D images where the [D, H, W]
            batch_size (int): The size of the batches for processing. If None, processes full stack at once.
            sigmoid (bool): Apply sigmoid trasnform to the output.
            **kwargs: Additional keyword arguments to pass to the SlidingWindowInferer.

        Returns:
            numpy.ndarray: The segmentation mask or stack of segmentation masks.
        """
        # TODO: Implement efficient batch size inference
        # Prepare image
        image, added_dim_index = self.ensure_4d(image, is_3D)
        img_tensor = torch.from_numpy(image).to(self.device)
        if self.scale_method != 'none':
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
            output_mask = inferer(img_tensor, self.model)
            if sigmoid:
                output_mask = torch.sigmoid(output_mask)
            output_mask = output_mask.cpu().numpy()

        if added_dim_index is not None:
            output_mask = np.squeeze(output_mask, axis=added_dim_index)

        # Free up tensors
        del img_tensor, image
        self.cleanup()
        
        return output_mask