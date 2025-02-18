# Standard library imports
import gc
import json
import os
import sys

# Third-party library imports
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit

# Local imports
from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.utils import get_device

# Type hints
from typing import Any, Dict, List, Optional, Tuple, Union
    
class CellClassifier(BaseModel):
    """
    A class for classifying cells in microscopy images.

    This classifier uses a pre-trained neural network model to classify regions of interest (ROIs)
    in labeled microscopy images. It supports batch processing and can handle various input sizes.

    Attributes:
        batch_size (int): The number of ROIs to process in each batch.
        min_size (Optional[int]): The minimum size for ROIs. Smaller ROIs will be resized.
        device (torch.device): The device (CPU, CUDA-GPU, or MPS) to use for computations.
        classifier (torch.nn.Module): The neural network model used for classification.
        classes (Optional[Dict[int, str]]): A mapping of class indices to class names.

    Methods:
        classify_rois: Classify ROIs in a labeled image.
        extract_rois: Extract ROIs from a labeled image.
        preprocess_rois: Preprocess ROIs for classification.
        get_model: Load a pre-trained model.
    """
    def __init__(self, model_path: str, 
                model_graph: torch.nn.Module, 
                min_size: Optional[int] = None,
                device: torch.device = get_device(), 
                classes: Optional[Dict[int, str]] = None,
                batch_size: int = 1024):        
        # Safety checks
        assert min_size is None or (isinstance(min_size, int) and min_size > 0), "min_size must be None or a positive integer"
        assert isinstance(device, torch.device), "device must be a valid torch device"
        assert classes is None or isinstance(classes, dict), "classes must be None or a dictionary"
        assert os.path.exists(model_path), f"Model file not found at path: {model_path}"

        # Load attributes
        self.batch_size = batch_size
        self.min_size = min_size
        self.device=device
        self.classifier = self.get_model(model_path, model_graph, device)
        self.classes = classes

    def classify_rois(self, 
                    labeled_image: np.ndarray, 
                    source_image: np.ndarray,
                    output_type: str = 'class-str') -> Tuple[Union[np.ndarray, List[str]], np.ndarray]:
        """
        Classify regions of interest (ROIs) in the given images.

        Args:
            labeled_image (np.ndarray): The labeled image containing ROIs.
            source_image (np.ndarray): The source image.
            output_type (str): The type of output ('class-str' or 'class-index').

        Returns:
            Tuple[Union[np.ndarray, List[str]], np.ndarray]: Classifications and labels.

        Raises:
            ValueError: If preprocessed_rois is empty or output_type is invalid.
        """

        # Preprocess ROIs
        rois = self.extract_rois(labeled_image, source_image, self.min_size)
        preprocessed_rois, labels = self.preprocess_rois(rois)

        if preprocessed_rois.numel() == 0:
            raise ValueError("preprocessed_rois is empty")
        
        # Classify objects in batchest to prevent memory overflow
        classifications=[]
        self.classifier.eval()
        with torch.no_grad():
            for i in range(0, len(preprocessed_rois), self.batch_size):
                batch = preprocessed_rois[i:i+self.batch_size]
                batch_classifications = self.classifier(batch)
                classifications.append(batch_classifications)
            
            classifications = torch.cat(classifications, dim=0)
            classifications = torch.softmax(classifications, dim = 1)
            class_indices = torch.argmax(classifications, dim=1)

        # Return class name or index
        if output_type == 'class-str':
            class_indices = class_indices.cpu().numpy()
            max_class_index = max(self.classes.keys())
            class_names_array = np.array([f"Unknown_Class_{i}" for i in range(max_class_index + 1)], dtype=object)
            for idx, name in self.classes.items():
                class_names_array[idx] = name
            class_names = class_names_array[class_indices]
        else:
            class_indices = class_indices.cpu().numpy()
        
        del preprocessed_rois, labeled_image, source_image, classifications

        self.cleanup()
        
        return (class_names, labels) if output_type == 'class-str' else (class_indices, labels)
    
    def preprocess_rois(self, rois):
        """Pre-process the ROIs images"""
        # Dict -> Numpy stack -> 4D (B, C, H, W) stack -> Tensor
        preprocessed_rois, labels = self.dict_to_numpy_stack(rois)
        preprocessed_rois, _= self.ensure_4d(preprocessed_rois, False)
        preprocessed_rois=torch.tensor(preprocessed_rois).float().to(self.device)
        return preprocessed_rois, labels

    def extract_rois(self, labeled_image, source_image, min_size=None):
        """Crop ROIs from the image and make uniform size for classification"""

        assert labeled_image.shape == source_image.shape, "Images must have the same shape"
        rois = {}
        for label, slices in enumerate(ndimage.find_objects(labeled_image), start=1):
            if slices is not None:
                cropped_source = source_image[slices]
                cropped_mask = (labeled_image[slices] == label)
                masked_roi = np.where(cropped_mask, cropped_source, 0)
                
                if min_size is not None:
                    masked_roi = self.resize_roi(masked_roi, min_size)

                rois[label] = masked_roi
        return rois

    @staticmethod
    def resize_roi(image, min_size):
        """Function to make ROIs of uniform size"""
        # Check if the image is 3D (Z, H, W)
        is_3d = image.ndim == 3
        
        if is_3d:
            image = image[0]

        h, w = image.shape
        
        if h < min_size and w < min_size:
            # Pad the image
            pad_height = max(0, min_size - h)
            pad_width = max(0, min_size - w)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        else:
            # Resize the image
            scale = min_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = ndimage.zoom(image, (new_h / h, new_w / w), order=1)
            
            # Pad if necessary after resizing
            if new_h < min_size or new_w < min_size:
                pad_h = max(0, min_size - new_h)
                pad_w = max(0, min_size - new_w)
                resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode='constant')
            
            return resized

    @staticmethod
    def dict_to_numpy_stack(roi_dict):
        """Efficient transformation of the numpy dict of ROIs to numpy stack"""

        # Pre-allocate memory based on the shape of first object and dict lenght and fill
        first_roi = next(iter(roi_dict.values()))
        roi_shape = first_roi.shape
        stack = np.zeros((len(roi_dict),) + roi_shape, dtype=first_roi.dtype)
        labels = np.zeros(len(roi_dict), dtype=int)
        for i, (label, roi) in enumerate(roi_dict.items()):
            stack[i] = roi
            labels[i] = label
        
        return stack, labels


class FlexResNet(nn.Module):
    """
    Flexible framework to build neural network for image classification based on Residual Blocks.

    Args:
        num_classes (int): Number of output classes.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        min_size (int, optional): Minimum input size. Defaults to 32.
        initial_filters (int, optional): Number of filters in the first layer. Defaults to 64.
        num_blocks (int, optional): Number of residual blocks. Defaults to 2.
        use_global_pool (bool, optional): Whether to use global average pooling. Defaults to True.
        **residual_unit_kwargs: Additional keyword arguments for ResidualUnit.

    Raises:
        ValueError: If invalid arguments are provided.
    """
    def __init__(self, 
                num_classes, 
                in_channels=1, 
                min_size=32, 
                initial_filters=64, 
                num_blocks=2, 
                use_global_pool=True,
                **residual_unit_kwargs):
        super().__init__()
        self.min_size = min_size

        layers = []
        current_channels = in_channels

        for i in range(num_blocks):
            out_channels = initial_filters * (2**i)
            layers.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=current_channels,
                    out_channels=out_channels,
                    strides=2 if i == 0 else 1,
                    kernel_size=3,
                    subunits=2 if i == 0 else 3,
                    **residual_unit_kwargs
                )
            )
            current_channels = out_channels

        if use_global_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            layers.append(nn.Flatten())

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # Pad if necessary
        if x.size(2) < self.min_size or x.size(3) < self.min_size:
            pad_h = max(0, self.min_size - x.size(2))
            pad_w = max(0, self.min_size - x.size(3))
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x