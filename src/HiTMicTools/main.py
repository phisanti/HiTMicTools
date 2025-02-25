import os
import sys
sys.path.insert(0, './src')

from monai.networks.nets import UNet as monai_unet
from HiTMicTools.utils import read_metadata
from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.toprak_updated_nn import Toprak_updated_nn
from HiTMicTools.pipelines.ASCT_focusrestore import ASCT_focusRestoration
from HiTMicTools.model_arch.nafnet import NAFNet
from HiTMicTools.model_arch.flexresnet import FlexResNet

def main(config_file: str, worklist: str = None):

    if worklist is None:
        worklist_id = ""
    else:
        worklist_id = os.path.basename(worklist).split(".")[0]
        print(f'Using worklist: {worklist}')

    # Read and parse a YAML file
    c_reader = ConfReader(config_file)
    configs = c_reader.opt
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

    analysis_wf.load_model('segmentator2', 
                           configs.segmentation["model_path"],
                           monai_unet(**segmentator_args['model_args']),
                           **configs.segmentation["inferer_args"]
                           )
    analysis_wf.load_model('cell-classifier', 
                           configs.cell_classifier["model_path"],
                           FlexResNet(**cell_classifier_args['model_args']),
                           **configs.cell_classifier["model_args"]
                           )

    analysis_wf.load_model('pi-classifier', 
                           configs.pi_classification["pi_classifier_path"])

    if pipeline_name == "ASCT_focusrestore":
        bf_focusrestore = read_metadata(configs.bf_focus["model_metadata"])
        fl_focusrestore = read_metadata(configs.fl_focus["model_metadata"])
        print('Loading focus restoration models')
        analysis_wf.load_model('focus-restorer-bf', 
                            configs.bf_focus["model_path"],
                            NAFNet(**bf_focusrestore['model_args']),
                            **configs.bf_focus["inferer_args"]
                            )
        analysis_wf.load_model('focus-restorer-fl', 
                            configs.fl_focus["model_path"],
                            NAFNet(**fl_focusrestore['model_args']),
                            **configs.fl_focus["inferer_args"]
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


if __name__ == "__main__":
    #import multiprocessing
    #multiprocessing.set_start_method('fork', force=True)


    if len(sys.argv) < 2:
        print("Usage: python main_final.py <config_file> [<worklist_file>]")
        sys.exit(1)

    config_file_path = sys.argv[1]
    worklist_id = ""

    if len(sys.argv) > 2:
        extra_arg = sys.argv[2]
        if os.path.isfile(extra_arg) and extra_arg.endswith(".txt"):
            c_reader = ConfReader(config_file_path)
            configs = c_reader.opt
            configs.input_data["file_list"] = extra_arg
            worklist_id = os.path.basename(extra_arg).split(".")[0]
        else:
            print("Invalid extra argument. Please provide a txt file or a folder.")
            sys.exit(1)

    main(config_file_path, worklist_id)
    #main()