import os
import sys
import unittest

from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.toprak_updated_nn import Toprak_updated_nn
from HiTMicTools.pipelines.ASCT_focusrestore import ASCT_focusRestoration

class TestPipelineConfigLoading(unittest.TestCase):
    def setUp(self):
        self.test_config = "./config/test_model_bundle.yml"
        self.pipeline_map = {
            "ASCT_focusrestore": ASCT_focusRestoration,
            "toprak_nn": Toprak_updated_nn,
        }

    def test_pipeline_config_loading(self):
        # 1. Test config file exists
        self.assertTrue(os.path.exists(self.test_config), f"Config file not found: {self.test_config}")

        # 2. Load configuration
        c_reader = ConfReader(self.test_config)
        configs = c_reader.opt

        # 3. Pipeline initialization
        pipeline_name = configs.pipeline_setup["name"]
        self.assertIn(pipeline_name, self.pipeline_map, f"Invalid pipeline name: {pipeline_name}")
        analysis_pipeline = self.pipeline_map[pipeline_name]
        analysis_wf = analysis_pipeline(
            configs.input_data["input_folder"],
            configs.input_data["output_folder"],
            file_type=configs.input_data["file_type"],
        )
        analysis_wf.load_config_dict(configs.pipeline_setup)

        # 4. Model loading
        model_bundle = configs.get("models", {}).get("model_collection")
        if model_bundle and os.path.exists(model_bundle):
            analysis_wf.load_model_bundle(model_bundle)
        # else: skip if not present

        # 5. Tracker loading
        if configs.pipeline_setup.get("tracking", False):
            tracking_config = configs.get("tracking", {})
            tracker_override_args = tracking_config.get("parameters_override", None)
            config_path = tracking_config.get("config_path")
            if model_bundle and os.path.exists(model_bundle):
                try:
                    analysis_wf.load_tracker(model_bundle, tracker_override_args=tracker_override_args)
                except Exception:
                    if config_path and os.path.exists(config_path):
                        analysis_wf.load_tracker(config_path, tracker_override_args=tracker_override_args)
            elif config_path and os.path.exists(config_path):
                analysis_wf.load_tracker(config_path, tracker_override_args=tracker_override_args)
            # else: skip if not present

if __name__ == "__main__":
    unittest.main()
