# Standard library imports
import os

# Third-party library imports
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

# Local imports
from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.resource_management.sysutils import empty_gpu_cache, get_device
from HiTMicTools.img_processing.img_ops import dynamic_resize_roi


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

    def __init__(
        self,
        model_path: str,
        model_graph: torch.nn.Module,
        min_size: Optional[int] = None,
        device: torch.device = get_device(),
        classes: Optional[Dict[int, str]] = None,
        batch_size: int = 1024,
        compile_mode: str = False,
    ):
        """
        Instantiate the CPU/memory-friendly classifier and load the serialized weights.

        Args:
            model_path: Path to the torch checkpoint on disk.
            model_graph: Model constructor used to build the network.
            min_size: Optional enforced ROI size prior to classification.
            device: Torch device (CPU/CUDA/MPS) the classifier runs on.
            classes: Optional mapping from numeric predictions to string labels.
            batch_size: Number of ROIs handled per inference step.
            compile_mode (str or False): Torch compile mode. Options:
                - "default": Fast compilation, good performance
                - "reduce-overhead": Optimized for small batches, uses CUDA graphs
                - "max-autotune": Slowest compilation, best runtime performance
                - False: Disable torch.compile entirely
        """
        # Safety checks
        assert min_size is None or (isinstance(min_size, int) and min_size > 0), (
            "min_size must be None or a positive integer"
        )
        assert isinstance(device, torch.device), "device must be a valid torch device"
        assert classes is None or isinstance(classes, dict), (
            "classes must be None or a dictionary"
        )
        assert os.path.exists(model_path), f"Model file not found at path: {model_path}"

        # Load attributes
        self.batch_size = batch_size
        self.min_size = min_size
        self.device = device
        self.classifier = self.get_model(model_path, model_graph, device, compile_mode=compile_mode)
        self.classes = classes

    def classify_rois(
        self,
        labeled_image: np.ndarray,
        source_image: np.ndarray,
        output_type: str = "class-str",
    ) -> Tuple[Union[np.ndarray, List[str]], np.ndarray]:
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
        classifications = []
        self.classifier.eval()
        self.classifier.to(self.device)
        with torch.no_grad():
            for i in range(0, len(preprocessed_rois), self.batch_size):
                batch = preprocessed_rois[i : i + self.batch_size]
                batch_classifications = self.classifier(batch)
                classifications.append(batch_classifications)

            classifications = torch.cat(classifications, dim=0)
            classifications = torch.softmax(classifications, dim=1)
            class_indices = torch.argmax(classifications, dim=1)

        # Return class name or index
        if output_type == "class-str":
            class_indices = class_indices.cpu().numpy()
            max_class_index = max(self.classes.keys())
            class_names_array = np.array(
                [f"Unknown_Class_{i}" for i in range(max_class_index + 1)], dtype=object
            )
            for idx, name in self.classes.items():
                class_names_array[idx] = name
            class_names = class_names_array[class_indices]
        else:
            class_indices = class_indices.cpu().numpy()

        del preprocessed_rois, labeled_image, source_image, classifications

        self.cleanup()
        labels = labels.cpu().numpy()
        return (
            (class_names, labels)
            if output_type == "class-str"
            else (class_indices, labels)
        )

    def preprocess_rois(self, rois):
        """Pre-process the ROIs images"""
        # Dict -> Torch stack -> 4D (B, C, H, W)
        preprocessed_rois, labels = self.dict_to_torch_stack(rois)
        preprocessed_rois, _ = self.ensure_4d(preprocessed_rois, False)
        return preprocessed_rois, labels

    def extract_rois(self, labeled_image, source_image, min_size=None):
        """Crop ROIs from the image and make uniform size for classification using torch-based resizing."""
        assert labeled_image.shape == source_image.shape, (
            "Images must have the same shape"
        )
        rois = {}
        for label, slices in enumerate(ndimage.find_objects(labeled_image), start=1):
            if slices is not None:
                cropped_source = source_image[slices]
                cropped_mask = labeled_image[slices] == label
                masked_roi = np.where(cropped_mask, cropped_source, 0)
                # Move to device as early as possible
                roi_tensor = torch.from_numpy(masked_roi).float().to(self.device)
                if min_size is not None:
                    roi_tensor = dynamic_resize_roi(roi_tensor, min_size)
                rois[label] = roi_tensor
        return rois

    @staticmethod
    def dict_to_torch_stack(roi_dict):
        """Efficient transformation of the dict of PyTorch ROIs to a torch stack"""
        # All tensors should already be on the same device from extract_rois_torch
        roi_list = [
            roi if isinstance(roi, torch.Tensor) else torch.from_numpy(roi).float()
            for roi in roi_dict.values()
        ]
        stack = torch.stack(roi_list, dim=0)
        labels = torch.tensor(list(roi_dict.keys()), dtype=torch.int64)
        # Note: we don't move labels to device as they're only used for output indexing
        return stack, labels
