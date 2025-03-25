import os
import sys
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.utils import read_metadata
from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.toprak_updated_nn import Toprak_updated_nn
from HiTMicTools.pipelines.ASCT_focusrestore import ASCT_focusRestoration
from HiTMicTools.model_arch.nafnet import NAFNet
from HiTMicTools.model_arch.flexresnet import FlexResNet


def build_and_run_pipeline(config_file: str, worklist: str = None):
    """
    Build and run the image analysis pipeline based on configuration.
    
    Args:
        config_file (str): Path to the configuration file
        worklist (str, optional): Path to the worklist file. Defaults to None.
    """
    # Initialize worklist_id and file_list
    worklist_id = None
    c_reader = ConfReader(config_file)
    configs = c_reader.opt
    
    if worklist is not None and worklist.strip():
        if not os.path.isfile(worklist):
            print(f"Error: Worklist file not found: {worklist}")
            sys.exit(1)
        worklist_id = os.path.basename(worklist).split(".")[0]
        print(f"Using worklist: {worklist}")
        # Update the config with the worklist
        configs.input_data["file_list"] = worklist
    else:
        # Explicitly set file_list to None when no worklist is provided
        configs.input_data["file_list"] = None

    extra_args = configs.get("extra", {})
    num_workers = configs.pipeline_setup.get("num_workers", {})

    pipeline_map = {
        "ASCT_focusrestore": ASCT_focusRestoration,
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

    analysis_wf.load_config_dict(configs.pipeline_setup)
    segmentator_args = read_metadata(configs.segmentation["model_metadata"])
    cell_classifier_args = read_metadata(configs.cell_classifier["model_metadata"])

    analysis_wf.load_model(
        "segmentator2",
        configs.segmentation["model_path"],
        monai_unet(**segmentator_args["model_args"]),
        **configs.segmentation["inferer_args"],
    )
    analysis_wf.load_model(
        "cell-classifier",
        configs.cell_classifier["model_path"],
        FlexResNet(**cell_classifier_args["model_args"]),
        **configs.cell_classifier["model_args"],
    )

    analysis_wf.load_model(
        "pi-classifier", configs.pi_classification["pi_classifier_path"]
    )

    if pipeline_name == "ASCT_focusrestore":
        bf_focusrestore = read_metadata(configs.bf_focus["model_metadata"])
        fl_focusrestore = read_metadata(configs.fl_focus["model_metadata"])
        print("Loading focus restoration models")
        analysis_wf.load_model(
            "focus-restorer-bf",
            configs.bf_focus["model_path"],
            NAFNet(**bf_focusrestore["model_args"]),
            **configs.bf_focus["inferer_args"],
        )
        analysis_wf.load_model(
            "focus-restorer-fl",
            configs.fl_focus["model_path"],
            NAFNet(**fl_focusrestore["model_args"]),
            **configs.fl_focus["inferer_args"],
        )
    else:
        print("Skipping focus restoration")

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
