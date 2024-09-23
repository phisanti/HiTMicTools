# Standard library imports
import gc
import json
import os
import sys

# Third-party library imports
import cupy as cp
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit

# Local imports
from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.utils import empty_gpu_cache, get_device

# Type hints
from typing import Any, Dict, List, Optional, Tuple, Union
    
class CellClassifier(BaseModel):
    """
    A class for classifying cells in microscopy images using CuPy analogue to the 
    HiTMicTools.model_components.cell_classifier version (CPU based)

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
                    labeled_image: cp.ndarray, 
                    source_image: cp.ndarray,
                    output_type: str = 'class-str') -> Tuple[Union[cp.ndarray, List[str]], cp.ndarray]:
        """
        Classify regions of interest (ROIs) in the given images.

        Args:
            labeled_image (cp.ndarray): The labeled image containing ROIs.
            source_image (cp.ndarray): The source image.
            output_type (str): The type of output ('class-str' or 'class-index').

        Returns:
            Tuple[Union[cp.ndarray, List[str]], cp.ndarray]: Classifications and labels.

        Raises:
            ValueError: If preprocessed_rois is empty or output_type is invalid.
        """

        # Preprocess ROIs
        rois = self.extract_rois(labeled_image, source_image, self.min_size)
        preprocessed_rois, labels = self.preprocess_rois(rois)

        if preprocessed_rois.size == 0:
            raise ValueError("preprocessed_rois is empty")
        
        # Classify objects in batches to prevent memory overflow
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
            class_names_array = cp.array([f"Unknown_Class_{i}" for i in range(max_class_index + 1)], dtype=object)
            for idx, name in self.classes.items():
                class_names_array[idx] = name
            class_names = class_names_array[class_indices]
        else:
            class_indices = class_indices.cpu().numpy()
        
        del preprocessed_rois, labeled_image, source_image, classifications
        gc.collect()
        empty_gpu_cache(self.device)
        
        return (class_names, labels) if output_type == 'class-str' else (class_indices, labels)
    
    def preprocess_rois(self, rois):
        """Pre-process the ROIs images"""
        # Dict -> CuPy stack -> 4D (B, C, H, W) stack -> Tensor
        preprocessed_rois, labels = self.dict_to_cupy_stack(rois)
        preprocessed_rois, _= self.ensure_4d(preprocessed_rois, False)
        preprocessed_rois=torch.tensor(preprocessed_rois).float().to(self.device)
        return preprocessed_rois, labels

    def extract_rois(self, labeled_image, source_image, min_size=None):
        """Crop ROIs from the image and make uniform size for classification"""

        assert labeled_image.shape == source_image.shape, "Images must have the same shape"
        rois = {}
        for label, slices in enumerate(ndimage.find_objects(cp.asnumpy(labeled_image)), start=1):
            if slices is not None:
                cropped_source = source_image[slices]
                cropped_mask = (labeled_image[slices] == label)
                masked_roi = cp.where(cropped_mask, cropped_source, 0)
                
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
            
            return cp.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        else:
            # Resize the image
            scale = min_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = ndimage.zoom(cp.asnumpy(image), (new_h / h, new_w / w), order=1)
            resized = cp.array(resized)
            
            # Pad if necessary after resizing
            if new_h < min_size or new_w < min_size:
                pad_h = max(0, min_size - new_h)
                pad_w = max(0, min_size - new_w)
                resized = cp.pad(resized, ((0, pad_h), (0, pad_w)), mode='constant')
            
            return resized

    @staticmethod
    def dict_to_cupy_stack(roi_dict):
        """Efficient transformation of the CuPy dict of ROIs to CuPy stack"""

        # Pre-allocate memory based on the shape of first object and dict length and fill
        first_roi = next(iter(roi_dict.values()))
        roi_shape = first_roi.shape
        stack = cp.zeros((len(roi_dict),) + roi_shape, dtype=first_roi.dtype)
        labels = cp.zeros(len(roi_dict), dtype=int)
        for i, (label, roi) in enumerate(roi_dict.items()):
            stack[i] = roi
            labels[i] = label
        
        return stack, labels