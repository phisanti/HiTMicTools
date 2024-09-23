import torch
import numpy as np
from typing import Tuple, Optional

class BaseModel:
    """
    A base class providing utility methods for loading models and ensuring consistent image dimensions.

    This class serves as a base for other model classes, offering common functionality such as
    loading pre-trained models and ensuring that input images have the expected 4-dimensional shape
    (batch, channel, height, width) required by PyTorch models.

    Methods:
        get_model(path, model_graph=None, device=None): Loads a model from the specified path.
        ensure_4d(img, is_3D): Ensures that the input image has 4 dimensions.
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
    def ensure_4d(
        img: np.ndarray, is_3D: bool
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
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