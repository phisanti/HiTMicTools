"""Model bundle creation and management utilities.

This module provides functionality to create model collection bundles (ZIP archives)
containing model weights, metadata, and configuration files for HiTMicTools pipelines.
"""

import os
import shutil
import yaml
import zipfile
import json
import datetime
from pathlib import Path
from typing import Optional


def create_model_bundle(
    models_info_path: str,
    output_bundle_path: str,
    auto_date: bool = True
) -> str:
    """Create a model bundle from a models info YAML file.

    This function reads a YAML configuration describing model paths and metadata,
    then packages everything into a ZIP archive with standardized naming and structure.

    Args:
        models_info_path: Path to the YAML file describing models to bundle
        output_bundle_path: Path where the output ZIP bundle will be saved
        auto_date: If True, automatically insert current date into filename (default: True)

    Returns:
        str: The actual path where the bundle was created (may differ from input if auto_date=True)

    Raises:
        FileNotFoundError: If models_info_path doesn't exist
        ValueError: If output_bundle_path doesn't have .zip extension

    Example:
        >>> create_model_bundle(
        ...     "config/models_info.yml",
        ...     "my_bundle.zip"
        ... )
        'my_bundle_20251218.zip'
    """
    # Validate inputs
    if not os.path.exists(models_info_path):
        raise FileNotFoundError(f"Models info file not found: {models_info_path}")

    output_path = Path(output_bundle_path)
    if output_path.suffix != '.zip':
        raise ValueError(f"Output path must have .zip extension, got: {output_path.suffix}")

    # Auto-insert date into filename if requested
    if auto_date:
        date_str = datetime.date.today().strftime("%Y%m%d")
        # Insert date before .zip extension
        output_path = output_path.parent / f"{output_path.stem}_{date_str}.zip"

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the models info from the YAML file
    with open(models_info_path, 'r') as file:
        models_info = yaml.safe_load(file)

    # Create temporary directories for the bundle
    temp_dir = 'temp_model_bundle'
    models_dir = os.path.join(temp_dir, 'models')
    metadata_dir = os.path.join(temp_dir, 'metadata')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Define a mapping for simplified model and metadata names
    model_name_map = {
        'bf_focus': 'NAFNet-bf_focus_restoration',
        'fl_focus': 'NAFNet-fl_focus_restoration',
        'segmentation': 'MonaiUnet-segmentation',
        'cell_classifier': 'FlexResNet-cell_classifier',
        'pi_classification': 'PIClassifier',
        'oof_detector': 'RFDETR-oof_detector',
        'sc_segmenter': 'RFDETRSegm-sc_segmenter',
    }

    # Handle tracker configuration if present
    if 'tracker' in models_info and 'config_path' in models_info['tracker']:
        tracker_config_path = models_info['tracker']['config_path']
        if os.path.isfile(tracker_config_path):
            shutil.copy(tracker_config_path, os.path.join(temp_dir, 'config_tracker.yml'))
        else:
            print(f"Warning: Tracker config file not found: {tracker_config_path}")

    # Iterate over each model in the models info
    for model_key, model_data in models_info.items():
        # Skip tracker section as it's handled separately
        if model_key == 'tracker':
            continue

        if model_key in model_name_map:
            # Check if model_path exists in the model_data
            if 'model_path' in model_data:
                model_path = model_data['model_path']
                original_filename = os.path.basename(model_path)

                if model_key == 'pi_classification':
                    # Use the original file extension for PIClassifier
                    original_ext = os.path.splitext(model_path)[1]
                    model_name = f"{model_name_map[model_key]}{original_ext}"
                else:
                    model_name = f"{model_name_map[model_key]}.pth"
                shutil.copy(model_path, os.path.join(models_dir, model_name))
            else:
                print(f"Warning: 'model_path' not found for {model_key}")
                continue  # Skip if no model path

            # Handle metadata for all models
            metadata_name = f"{model_name_map[model_key]}.json"
            metadata_path = os.path.join(metadata_dir, metadata_name)
            metadata = {}

            # Always include original_name in metadata
            metadata['original_name'] = original_filename

            # If a metadata file exists, load and update it
            if 'model_metadata' in model_data and os.path.isfile(model_data['model_metadata']):
                with open(model_data['model_metadata'], 'r') as f:
                    loaded_metadata = json.load(f)
                loaded_metadata['original_name'] = original_filename
                metadata = loaded_metadata

            # Write metadata file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    # Write the inner config.yml file with creation metadata
    config = {
        '_bundle_metadata': {
            'creation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_config': os.path.basename(models_info_path),
        }
    }

    for model_key, model_data in models_info.items():
        # Skip tracker section in main config
        if model_key == 'tracker':
            continue

        if model_key in model_name_map:
            config[model_key] = {}
            if 'model_path' in model_data:
                if model_key == 'pi_classification':
                    # Use the original file extension for PIClassifier
                    original_ext = os.path.splitext(model_data['model_path'])[1]
                    model_filename = f"{model_name_map[model_key]}{original_ext}"
                    config[model_key]['model_path'] = f"models/{model_filename}"
                else:
                    config[model_key]['model_path'] = f"models/{model_name_map[model_key]}.pth"

            # Add metadata path for all models
            config[model_key]['model_metadata'] = f"metadata/{model_name_map[model_key]}.json"

            if 'inferer_args' in model_data:
                config[model_key]['inferer_args'] = model_data['inferer_args']
            if 'model_args' in model_data:
                config[model_key]['model_args'] = model_data['model_args']

    with open(os.path.join(temp_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    # Create the zip bundle
    output_path_str = str(output_path)
    with zipfile.ZipFile(output_path_str, 'w', zipfile.ZIP_DEFLATED) as bundle_zip:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                bundle_zip.write(file_path, os.path.relpath(file_path, temp_dir))

    # Clean up temporary directories
    shutil.rmtree(temp_dir)

    return output_path_str
