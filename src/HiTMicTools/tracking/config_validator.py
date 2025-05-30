import numpy as np
from typing import Dict, Any


class TrackingConfigValidator:
    """Validates tracking configuration matrices and parameters."""
    
    def validate_config_dimensions(self, config: Dict[str, Any]) -> bool:
        """
        Validate motion model matrix dimensions.
        
        Args:
            config: Configuration dictionary containing motion model
            
        Returns:
            True if all matrices have correct dimensions
            
        Raises:
            ValueError: If matrix dimensions are incorrect
        """
        motion = config["motion_model"]
        m, s = motion["measurements"], motion["states"]
        
        matrix_checks = {
            "A": (s * s, "State transition matrix"),
            "H": (m * s, "Observation matrix"), 
            "P": (s * s, "Covariance matrix"),
            "R": (m * m, "Measurement noise matrix"),
            "G": (s, "Process noise matrix")
        }
        
        for matrix_name, (expected_size, description) in matrix_checks.items():
            actual_size = len(motion[matrix_name]["matrix"])
            if actual_size != expected_size:
                raise ValueError(
                    f"{description} ({matrix_name}): expected {expected_size}, "
                    f"got {actual_size}"
                )
        
        return True
    
    def validate_features(self, features: list, available_columns: list) -> bool:
        """
        Validate that required features are available in DataFrame.
        
        Args:
            features: List of required feature names
            available_columns: List of available column names
            
        Returns:
            True if all features are available
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = [f for f in features if f not in available_columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return True
    
    def validate_tracking_data(self, df_columns: list) -> bool:
        """
        Validate that DataFrame has required columns for tracking.
        
        Args:
            df_columns: List of DataFrame column names
            
        Returns:
            True if required columns are present
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['t', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in df_columns]
        if missing_cols:
            raise ValueError(f"Missing required tracking columns: {missing_cols}")
        return True
