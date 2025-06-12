# HiTMicTools

A comprehensive toolkit for High-Throughput Microscopy Analysis developed by the Boeck Lab. This package provides tools and utilities for processing, analyzing, and managing large-scale microscopy data.

## 🎯 Features

- Image preprocessing and enhancement
- Region of Interest (ROI) analysis with both CPU and GPU support
- Batch processing capabilities for large datasets
- Configurable processing pipelines
- Memory-efficient operations with logging
- Command-line interface for easy automation

## 📋 Requirements
HiTMicTools requires Python 3.9 or later and depends on the following packages:

```
numpy
torch
torchvision
matplotlib
seaborn
pandas
scikit-learn
scikit-image
scipy
tifffile
monai
pystackreg
GPUtil
psutil
nd2
opencv-python
ome-types
pyyaml
joblib
jax==0.4.23
jaxlib==0.4.23 
basicpy
jetraw-tools
onnxruntime
skl2onnx
```
The fix versioning of the jax libraries is due to the basicpy package which is still under heavy development.

For CUDA support (optional):
```
cupy-cuda11x
cudf
cucim
```
Moreover, it is also necessary to install the jetraw-tools and basicpy package from the source:
```bash
   pip install git+https://github.com/yuliu96/BaSiCPy.git
   pip install git+https://github.com/phisanti/jetraw_tools.git
```

The `jetraw-tools` package also depends on the jetraw software and having a valid licence. This is only required if working with `.p.tiff` files.

## 🚀 Installation
We recommend to create a conda environment with python 3.9 for best compatibility with the dependencies.
```bash
conda create -n hitmictools python=3.9
conda activate hitmictools
```
Then, this project can be easily installed via pip from the repository:
```bash
pip install git+https://github.com/phisanti/HiTMicTools
```

However, if you would like to contribute or suggest any change, you can also clone the source:
```bash
git clone https://github.com/phisanti/HiTMicTools
cd HiTMicTools
pip install -e .
```
If you plan to use the tracking functionalities within HiTMicTools, you will need to install `btrack`. Due to potential dependency conflicts with `ome-types` when installing `btrack` via `pip install btrack`, and issues with C++ library compilation when installing directly from git using `pip install git+btrack/btrack at main · quantumjot/btrack`, we recommend manual compilation and installation of `btrack` for full functionality.

1.  **Clone the btrack repository:**
    ```bash
    git clone https://github.com/quantumjot/btrack.git
    ```
2.  **Navigate to the btrack directory:**
    ```bash
    cd btrack
    ```
3.  **Compile btrack:**
    `btrack` includes a `build.sh` script to simplify the compilation process.
    ```bash
    bash build.sh
    ```
4.  **Install btrack:**
    After successful compilation, install the package.
    ```bash
    pip install .
    ```
5.  **Clean up (optional):**
    You can optionally clean up the build artifacts.
    ```bash
    python setup.py clean --all
    ```
    Navigate back to your previous directory if needed:
    ```bash
    cd ..
    ```
## 📖 Usage

### Command Line Interface

The package provides a CLI for common operations so that the image analysis can be automated and streamlined. The basic operation is to run an image analysis pipeline configured with a configuration file. 

```bash
# Basic usage
hitmictools --help

# Run image analysis with a configuration file
hitmictools run --config <config_file> --worklist <worklist_file> # This second argument is optional
```

Also, the package provides a command to split a folder of images into smaller batches. This can be useful for processing large datasets in smaller chunks.
```bash
# Process a batch of images
hitmictools split-files --target-folder <folder> --n-blocks <num_blocks>
```

The last command is `generate-slurm`, which generates a SLURM script for running the analysis on a cluster. This command is particularly useful for large-scale processing tasks:
```bash
# Generate a bash script for SLURM
generate-slurm --job-name 'testing' --file-blocks --n-blocks 2 --conda-env 'img_analysis'  --config-file './your_config.yml'    
```

## 🔧 Configuration

The toolkit uses configuration files to customize processing parameters. Create a YAML configuration file:

```yaml
# Example configuration for ASCT focus restoration pipeline

# Pipeline type
pipeline_type: ASCT_focusRestoration

# Input/Output settings
input_dir: /path/to/input/directory
output_dir: /path/to/output/directory
file_type: nd2  # Supported formats: nd2, tif, tiff, etc.

# Image processing parameters
reference_channel: 0  # Brightfield channel index for segmentation
pi_channel: 1  # Fluorescence channel index for PI classification or other measurements
align_frames: true  # Whether to align frames in the stack
method: basicpy_fl  # Background removal method: 'standard', 'basicpy', or 'basicpy_fl'

# Model paths
bf_focus_restorer:
  model_path: /path/to/models/bf_focus_restoration_model.pth
  model_type: monai  # Model architecture type

fl_focus_restorer:
  model_path: /path/to/models/fl_focus_restoration_model.pth
  model_type: monai

image_segmentator:
  model_path: /path/to/models/segmentation_model.pth
  model_type: monai

object_classifier:
  model_path: /path/to/models/object_classifier.onnx
  model_type: onnx
  
# Optional PI classifier
pi_classifier:
  model_path: /path/to/models/pi_classifier.joblib
  model_type: sklearn

# Processing options
export_labeled_mask: true  # Export labeled mask images
export_aligned_image: true  # Export aligned images
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project has been developed by the Boeck Lab at the Univeristy Hospital of Basel. The code released here is under the European Union Public Licence 1.2.