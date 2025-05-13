import os
import sys
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.utils import read_metadata
from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.toprak_updated_nn import Toprak_updated_nn
from HiTMicTools.pipelines.ASCT_focusrestore import ASCT_focusRestoration


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
        print(f"Using worklist: {worklist}")
        # No need to extract worklist_id or update configs here
    else:
        # Set worklist to None explicitly when not provided
        worklist = None

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
        worklist_path=worklist,
    )

    analysis_wf.load_config_dict(configs.pipeline_setup)
    model_bundle = configs.get("models", {}).get("model_collection")
    if model_bundle:
        # Use the load_model_bundle method if a bundle is provided
        analysis_wf.load_model_bundle(model_bundle)
    else:
        # Load models individually using load_model_fromdict
        for model_key in [
            "segmentation",
            "cell_classifier",
            "bf_focus",
            "fl_focus",
            "pi_classification",
        ]:
            if model_key in configs:
                analysis_wf.load_model_fromdict(model_key, configs[model_key])

    if configs.pipeline_setup["parallel_processing"]:
        analysis_wf.process_folder_parallel(
            files_pattern=configs.input_data["file_pattern"],
            export_labeled_mask=configs.input_data["export_labelled_masks"],
            export_aligned_image=configs.input_data["export_labelled_masks"],
            num_workers=num_workers,
            **extra_args,
        )
    else:
        analysis_wf.process_folder(
            files_pattern=configs.input_data["file_pattern"],
            export_labeled_mask=configs.input_data["export_labelled_masks"],
            export_aligned_image=configs.input_data["export_labelled_masks"],
            **extra_args,
        )
