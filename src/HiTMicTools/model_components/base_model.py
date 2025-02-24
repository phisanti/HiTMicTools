import torch
import numpy as np
from typing import Tuple, Optional, Union, Any
from HiTMicTools.utils import empty_gpu_cache, get_device
import gc

class BaseModel:
    """
    A base class providing utility methods for loading models and ensuring consistent image dimensions.

    This class serves as a base for other model classes, offering common functionality such as
    loading pre-trained models and ensuring that input images have the expected 4-dimensional shape
    (batch, channel, height, width) required by PyTorch models.

    Methods:
        get_model(path, model_graph=None, device=None): Loads a model from the specified path.
        ensure_4d(img, is_3D): Ensures that the input image has 4 dimensions.
        cleanup(): Frees GPU memory and performs garbage collection.
    """

    def get_model(
        self, 
        path: str, 
        model_graph: torch.nn.Module = None, 
        device: torch.device = None,
    ) -> torch.nn.Module:
        """
        Loads a model from the specified path and returns it.

        Args:
            path (str): The path to the model file.
            device (str): The device to load the model onto.
            model_graph (Optional[torch.nn.Module]): An optional pre-initialized model graph.

        Returns:
            torch.nn.Module: The loaded model.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            RuntimeError: If an error occurs while loading the model.
            Exception: If an unexpected error occurs.
        """

        if device is None:
            device = get_device()

        try:
            if model_graph is None:
                model = torch.load(path, map_location=device)
            else:
                model = model_graph
                state_dict = torch.load(path, map_location=device)

                # Check if the loaded state dictionary is compatible with the model architecture
                if not set(state_dict.keys()).issubset(set(model.state_dict().keys())):
                    raise ValueError(
                        "Loaded state dictionary does not match the model architecture."
                    )

                model.load_state_dict(state_dict)

            model.to(device)
            torch.compile(model, mode="max-autotune")

            return model

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at path: {path}")

        except RuntimeError as e:
            raise RuntimeError(f"Error occurred while loading the model: {str(e)}")

        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")

    @staticmethod
    def ensure_4d(img: Union[np.ndarray, torch.Tensor], 
                  is_3D: bool,
                  dimension: Optional[Tuple[int, ...]] = None) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Tuple[int, ...]]]:
        """
        Ensures that the input image has 4 dimensions (batch, channel, height, width).

        Args:
            img (np.ndarray or torch.Tensor): The input image.
            is_3D (bool): Whether the image is a 3D volume or a 2D image.

        Returns:
            Tuple[np.ndarray or torch.Tensor, Optional[Tuple[int, int]]]: A tuple containing the image with 4 dimensions and a tuple of the indexes of the added dimensions (or None if no dimensions were added).
        """
        if isinstance(img, np.ndarray):
            return BaseModel.__ensure_4d_np(img, is_3D, dimension)
        elif isinstance(img, torch.Tensor):
            return BaseModel.__ensure_4d_torch(img, is_3D, dimension)
        else:
            raise TypeError(f"Unsupported data type: {type(img)}")


    @staticmethod
    def __ensure_4d_np(img: np.ndarray, 
                       is_3D: bool,
                       dimension: Optional[Tuple[int, ...]] = None
                       ) -> Tuple[np.ndarray, Optional[Tuple[int, ...]]]:
        """
        Ensures that the input image has 4 dimensions (batch, channel, height, width).
        This is the standard format expected by PyTorch models.

        Args:
            img (numpy.ndarray): The input image.
            is_3D (bool): Wether the image is a 3D volume or a 2D image.

        Returns:
        Tuple[numpy.ndarray, Optional[Tuple[int, int]]]: A tuple containing the image with 4 dimensions and a tuple of the indexes of the added dimensions (or None if no dimensions were added).
        """
        added_dim_indexes = None
        if dimension is not None:
            img = np.expand_dims(img, axis=dimension)
            if isinstance(dimension, tuple):
                added_dim_indexes = dimension
            else:
                added_dim_indexes = (dimension,)

            return img, added_dim_indexes
        if img.ndim == 2:
            # Add channel and batch dimensions if the array is 2D
            img = np.expand_dims(img, axis=(0, 1))
            added_dim_indexes = (0, 1)
        elif img.ndim == 3 and is_3D:
            # Add a channel dimension if the array is multi-stack
            img = np.expand_dims(img, axis=0)
            added_dim_indexes = (0,)
        elif img.ndim == 3 and not is_3D:
            # Add a batch dimension if the array is multi-channel
            img = np.expand_dims(img, axis=1)
            added_dim_indexes = (1,)
        elif img.ndim == 4:
            # No modification needed if the array is already 4D
            pass
        else:
            raise ValueError(
                f"Unsupported array dimensions: {img.ndim}, current shape is {img.shape}"
            )

        return img, added_dim_indexes

    @staticmethod
    def __ensure_4d_torch(img: torch.Tensor, 
                          is_3D: bool,
                          dimension: Optional[Tuple[int, ...]] = None
                          ) -> Tuple[torch.Tensor, Optional[Tuple[int, ...]]]:
        """
        Ensures that the input image has 4 dimensions (batch, channel, height, width).
        This is the standard format expected by PyTorch models.
        
        Args:
            img (torch.Tensor): The input image tensor.
            is_3D (bool): Whether the image is a 3D volume or a 2D image.
        
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[int, int]]]: A tuple containing the image tensor with 4 dimensions and a tuple of the indexes of the added dimensions (or None if no dimensions were added).
        """
        added_dim_indexes = None
        if dimension is not None:
            for dim in sorted(dimension):
                img = img.unsqueeze(dim)
                added_dim_indexes = dimension
            return img, added_dim_indexes
        if img.ndim == 2:
            # Add channel and batch dimensions if the tensor is 2D
            img = img.unsqueeze(0).unsqueeze(0)
            added_dim_indexes = (0, 1)
        elif img.ndim == 3 and is_3D:
            # Add a channel dimension if the tensor is multi-stack
            img = img.unsqueeze(0)
            added_dim_indexes = (0,)
        elif img.ndim == 3 and not is_3D:
            # Add a batch dimension if the tensor is multi-channel
            img = img.unsqueeze(1)
            added_dim_indexes = (1,)
        elif img.ndim == 4:
            # No modification needed if the tensor is already 4D
            pass
        else:
            raise ValueError(
                f"Unsupported tensor dimensions: {img.ndim}, current shape is {img.shape}"
            )
        
        return img, added_dim_indexes

    @staticmethod
    def restore_dims(img: Union[np.ndarray, torch.Tensor], 
                    added_dim_indexes: Optional[Tuple[int, ...]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Restores the original dimensions of the input image by removing the added dimensions.

        Args:
            img (np.ndarray or torch.Tensor): The input image.
            added_dim_indexes (Optional[Tuple[int, ...]]): The indexes of the added dimensions.

        Returns:
            Union[np.ndarray, torch.Tensor]: The image with the original dimensions restored.
        """
        if added_dim_indexes is None:
            return img

        if isinstance(img, np.ndarray):
            for dim in sorted(added_dim_indexes, reverse=True):
                img = np.squeeze(img, axis=dim)
        elif isinstance(img, torch.Tensor):
            for dim in sorted(added_dim_indexes, reverse=True):
                img = img.squeeze(dim)
        else:
            raise TypeError(f"Unsupported data type: {type(img)}")

        return img

    def cleanup(self):
        gc.collect()
        empty_gpu_cache(self.device)
