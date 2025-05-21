import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Union, Any


class PIClassifier:
    """
    Wrapper class for PI classification. This class supports both sklearn (joblib/pickle)
    and ONNX formats with a unified API.
    """

    def __init__(self, model_path: str):
        """
        Initialize the PI classifier wrapper.

        Args:
            model_path (str): Path to either a joblib/pickle model (.pkl/.joblib) or an ONNX model (.onnx)

        Raises:
            ValueError: If the model file has an unsupported extension or cannot be loaded
        """
        self.model_path = model_path
        self.model_type = self._determine_model_type(model_path)
        self.model = self._load_model(model_path, self.model_type)
        self.feature_names_in_ = self._get_feature_names()

    def _determine_model_type(self, model_path: str) -> str:
        """
        Determine model type based on file extension.

        Args:
            model_path (str): Path to the model file

        Returns:
            str: Model type, either 'sklearn' or 'onnx'

        Raises:
            ValueError: If the file extension is not supported (.pkl, .joblib, or .onnx)
        """
        _, ext = os.path.splitext(model_path)
        if ext.lower() in [".pkl", ".joblib"]:
            return "sklearn"
        elif ext.lower() == ".onnx":
            return "onnx"
        else:
            raise ValueError(
                f"Unsupported model file extension: {ext}. Expected .pkl, .joblib, or .onnx"
            )

    def _load_model(self, model_path: str, model_type: str) -> Any:
        """
        Load the model based on its type.

        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model to load ('sklearn' or 'onnx')

        Returns:
            Any: Loaded model object (either sklearn model or ONNX inference session)

        Raises:
            ValueError: If the model_type is not supported
            FileNotFoundError: If the model file doesn't exist
            ImportError: If onnxruntime is not installed (for ONNX models)
        """
        if model_type == "sklearn":
            with open(model_path, "rb") as f:
                return joblib.load(f)
        elif model_type == "onnx":
            import onnxruntime as ort

            return ort.InferenceSession(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _get_feature_names(self) -> List[str]:
        """
        Extract feature names from the model.

        For sklearn models, retrieves feature_names_in_ attribute.
        For ONNX models, attempts to extract feature names from model metadata.

        Returns:
            List[str]: List of feature names used by the model.
                       Returns empty list if feature names cannot be determined.
        """
        if self.model_type == "sklearn":
            return self.model.feature_names_in_
        elif self.model_type == "onnx":
            try:
                input_name = self.model.get_inputs()[0].name
                # If the input name looks like a tensor name and not a feature list,
                # return empty list (will need to be specified when predicting)
                if input_name in ["input", "X", "input_0", "input.1"]:
                    return []
                # Extract feature names if stored in metadata
                metadata = self.model.get_modelmeta().custom_metadata_map
                if "feature_names" in metadata:
                    return metadata["feature_names"].split(",")
            except (AttributeError, IndexError, KeyError):
                pass
            return []

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict classes using the loaded model.

        This method maintains consistent output format regardless of model type (sklearn or ONNX),
        always returning class labels as strings ('piNEG' or 'piPOS').

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input features as numpy array or pandas DataFrame.
                                                For ONNX models with known feature names, DataFrame columns
                                                will be ordered according to feature_names_in_.

        Returns:
            np.ndarray: Predicted classes as numpy array with values 'piNEG' or 'piPOS'

        Raises:
            ValueError: If the model type is not supported
            RuntimeError: If prediction with the ONNX model fails
            KeyError: If DataFrame is missing required columns for ONNX model
        """
        if self.model_type == "sklearn":
            return self.model.predict(X)
        elif self.model_type == "onnx":
            # Convert pandas DataFrame to numpy if necessary
            if isinstance(X, pd.DataFrame):
                # Ensure correct feature order for ONNX
                if self.feature_names_in_:
                    X = X[self.feature_names_in_].to_numpy().astype(np.float32)
                else:
                    X = X.to_numpy().astype(np.float32)
            elif isinstance(X, np.ndarray):
                X = X.astype(np.float32)

            # Get input name from the model
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name

            # Run inference
            results = self.model.run([output_name], {input_name: X})
            predictions = results[0]

            # Convert numeric predictions to class labels if necessary
            # Assuming binary classification with 0 = "piNEG" and 1 = "piPOS"
            if predictions.dtype in [np.int64, np.int32, np.float32, np.float64]:
                class_labels = np.array(["piNEG", "piPOS"])
                return class_labels[predictions.astype(int)]

            return predictions
