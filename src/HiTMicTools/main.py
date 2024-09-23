import os
import sys
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.model_components.cell_classifier import FlexResNet
from HiTMicTools.utils import read_metadata
from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.toprak_updated_nn import Toprak_updated_nn

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_final.py <config_file> [<worklist_file>]")
        sys.exit(1)

    config_file_path = sys.argv[1]
    worklist_id = ""

    if len(sys.argv) > 2:
        extra_arg = sys.argv[2]
        if os.path.isfile(extra_arg) and extra_arg.endswith(".txt"):
            configs.input_data["file_list"] = extra_arg
            worklist_id = os.path.basename(extra_arg).split(".")[0]
        else:
            print("Invalid extra argument. Please provide a txt file or a folder.")
            sys.exit(1)

    # Read and parse a YAML file
    c_reader = ConfReader(config_file_path)
    configs = c_reader.opt
    extra_args = configs.get("extra", {})
    num_workers = configs.pipeline_setup.get("num_workers", {})

    segmentator_args = read_metadata(configs.segmentation["model_metadata"])
    cell_classifier_args = read_metadata(configs.cell_classifier["model_metadata"])

    pipeline_map = {
        "toprak_nn": Toprak_updated_nn,
    }

    pipeline_name = configs.pipeline_setup["name"]
    analysis_pipeline = pipeline_map.get(pipeline_name)
    if analysis_pipeline is None:
        print(f"Invalid pipeline name: {pipeline_name}")
        sys.exit(1)

    analysis_wf = analysis_pipeline(
        configs.input_data["input_folder"],
        configs.input_data["output_folder"],
        file_type=configs.input_data["file_type"],
        worklist_id=worklist_id,
    )

    # Config image analysis
    analysis_wf.config_image_analysis(
        reference_channel=configs.pipeline_setup["reference_channel"],
        align_frames=configs.pipeline_setup["align_frames"],
        method=configs.pipeline_setup["method_clear_background"],
    )

    analysis_wf.load_model('segmentator', 
                           configs.segmentation["model_path"],
                           monai_unet(**segmentator_args['model_args']),
                           **configs.segmentation["segmentator_args"]
                           )
    analysis_wf.load_model('cell-classifier', 
                           configs.cell_classifier["model_path"],
                           FlexResNet(**cell_classifier_args['model_args']),
                           **configs.cell_classifier["cell_classifier_args"]
                           )

    analysis_wf.load_model('pi-classifier', 
                            configs.pi_classification["pi_classifier_path"])

    if configs.pipeline_setup["parallel_processing"]:
        analysis_wf.process_folder_parallel(
            files_pattern=configs.input_data["file_pattern"],
            file_list=configs.input_data["file_list"],
            export_labeled_mask=configs.input_data["export_labelled_masks"],
            export_aligned_image=configs.input_data["export_labelled_masks"],
            num_workers=num_workers,
            **extra_args,
        )
    else:
        analysis_wf.process_folder(
            files_pattern=configs.input_data["file_pattern"],
            file_list=configs.input_data["file_list"],
            export_labeled_mask=configs.input_data["export_labelled_masks"],
            export_aligned_image=configs.input_data["export_labelled_masks"],
            **extra_args,
        )

if __name__ == "__main__":
    main()