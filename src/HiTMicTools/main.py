import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.workflows import StandardAnalysis
from HiTMicTools.utils import read_metadata
from HiTMicTools.confreader import ConfReader


if len(sys.argv) < 2:
    print("Please pass the YAML configuration file as main.py /path/to/config.yml.")
    sys.exit(1)

# Read and parse a YAML file
config_file_path = sys.argv[1]
c_reader = ConfReader(config_file_path)
configs = c_reader.opt
configs

# Allow to dynamically override the input folder (useful for SLURM jobs)
if len(sys.argv) > 2:
    extra_arg = sys.argv[2]
    if os.path.isfile(extra_arg) and extra_arg.endswith(".txt"):
        configs.input_data["file_list"] = extra_arg
    else:
        print("Invalid extra argument. Please provide a txt file or a folder.")
        sys.exit(1)

model_metadata = read_metadata(configs.segmentation["model_metadata"])
seg_params = {
    "model_path": configs.segmentation["model_path"],
    "model_graph": monai_unet(**model_metadata["model_args"]),
    "scale_method": configs.segmentation["scale_method"],
    "patch_size": configs.segmentation["patch_size"],
    "overlap_ratio": configs.segmentation["overlap_ratio"],
    "half_precision": configs.segmentation["half_precision"],
}

# Instantiate analysis workflow
analysis_wf = StandardAnalysis(
    configs.input_data["input_folder"],
    configs.input_data["output_folder"],
    image_classifier_args=seg_params,
    object_classifier=configs.classification["object_classifier_path"],
    pi_classifier=configs.classification["pi_classifier_path"],
    file_type=configs.input_data["file_type"],
)

# Config image analysis
analysis_wf.config_image_analysis(
    reference_channel=configs.pre_processing["reference_channel"],
    align_frames=configs.pre_processing["align_frames"],
    method=configs.pre_processing["method_clear_background"],
)

# Start folder processing
if configs.input_data["parallel_processing"]:
    analysis_wf.process_folder_parallel(
        files_pattern=configs.input_data["file_pattern"],
        file_list=configs.input_data["file_list"],
        export_labeled_mask=True,
        export_aligned_image=True,
    )
else:
    analysis_wf.process_folder(
        files_pattern=configs.input_data["file_pattern"],
        file_list=configs.input_data["file_list"],
        export_labeled_mask=True,
        export_aligned_image=True,
    )
