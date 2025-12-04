import os
import glob
import time
import fnmatch
import zipfile
import tempfile
import yaml
import logging
from logging.handlers import MemoryHandler

# Resources imports
import concurrent.futures
import gc
import multiprocessing
from contextlib import contextmanager

# Type annotations and
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

# Local imports
from HiTMicTools import __version__
from HiTMicTools.resource_management.memlogger import MemoryLogger
from HiTMicTools.model_components.segmentation_model import Segmentator
from HiTMicTools.model_components.cell_classifier import CellClassifier
from HiTMicTools.model_components.focus_restorer import FocusRestorer
from HiTMicTools.model_components.oof_detector import OofDetector
from HiTMicTools.model_components.scsegmenter import ScSegmenter
from HiTMicTools.tracking.cell_tracker import CellTracker
from HiTMicTools.resource_management.sysutils import get_device, get_system_info
from HiTMicTools.model_arch.nafnet import NAFNet
from HiTMicTools.model_arch.flexresnet import FlexResNet
from HiTMicTools.model_components.pi_classifier import PIClassifier
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.utils import read_metadata, update_config


@contextmanager
def managed_resource(*objects):
    yield objects
    for obj in objects:
        del obj
    gc.collect()


class BasePipeline(ABC):
    """
    An abstract base class for performing standard analysis on microscopy images.

    This class provides the framework for image analysis pipelines but requires
    subclasses to implement the analyse_image method for specific analysis tasks.

    Methods:
        setup_logger: Set up a logger for logging the analysis progress.
        remove_logger: Remove logger, useful for concurrent parallel processing.
        load_model_fromdict: Load a model based on the specified model type and configuration dictionary.
        load_model_bundle: Load models and configurations from a bundled zip file.
        load_config_dict: Configure image analysis settings from a dictionary.
        config_image_analysis: Configure the image analysis settings.
        get_files: Retrieve a list of files from the specified input path, filtered by pattern and extension.
        process_folder: Process all files with the matching pattern and file extension in the input folder.
        process_folder_parallel: Process multiple image files in parallel using multiprocessing.
        analyse_image: (Abstract) Analyze a single image file. Must be implemented by subclasses.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        worklist_path: str = None,
        file_type: str = ".nd2",
    ):
        """Initialize the BasePipeline.

        Args:
            input_path (str): Path to the input directory containing the images.
            output_path (str): Path to the output directory for saving the analysis results.
            worklist_path (str, optional): Path to the worklist file. Defaults to None.
            file_type (str, optional): File extension of the image files. Defaults to '.nd2'.
        """
        self.input_path = input_path
        self.worklist_path = worklist_path
        last_folder = os.path.basename(os.path.normpath(self.input_path))

        worklist_id = ""
        if worklist_path:
            worklist_id = os.path.basename(worklist_path).split(".")[0]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.main_logger = self.setup_logger(
            output_path, last_folder, logger_id=worklist_id, print_output=True
        )

        self.output_path = output_path
        self.file_type = file_type

    def setup_logger(
        self,
        output_path: str,
        name: str,
        logger_id: str = "",
        print_output: bool = False,
    ) -> logging.Logger:
        """Set up a logger for logging the analysis progress."""

        # Set up logger file
        logging.setLoggerClass(MemoryLogger)
        last_folder = os.path.basename(os.path.normpath(name))
        log_file = os.path.join(output_path, f"{name}_{logger_id}_analysis.log")
        logger_name = f"{output_path}_{name}_{logger_id}"  # Use a unique identifier for each instance important for parallelisation
        logger = logging.getLogger(logger_name)

        # Set logger level and format
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Set logger for console if required
        if print_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def remove_logger(self, logger):
        """Remove logger, useful for concurrent parallel processing."""

        # Remove all handlers from the logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Delete the logger instance
        del logging.Logger.manager.loggerDict[logger.name]

    def load_model_fromdict(self, model_type: str, config_dic: Dict[str, Any]) -> None:
        """
        Load a model based on the specified model type and configuration dictionary.

        Args:
            model_type (str): Type of the model to load ('segmentator', 'segmentator2',
                            'cell-classifier', 'focus-restorer-fl', 'focus-restorer-bf', 'pi-classifier').
            config_dic (Dict[str, Any]): Dictionary containing model configuration including:
                            - 'model_path': Path to model weights file
                            - 'model_metadata': Path to model metadata (except for pi-classifier)
                            - 'inferer_args' or 'model_args': Additional configuration parameters

        Returns:
            None: The model is loaded and attached to the appropriate attribute in the class.

        Raises:
            ValueError: If an invalid model type is provided or required arguments are missing.
            KeyError: If required configuration keys are missing.
        """
        alias_map = {
            "segmentation": "segmentator",
            "cell_classifier": "cell-classifier",
            "bf_focus": "focus-restorer-bf",
            "fl_focus": "focus-restorer-fl",
            "pi_classification": "pi-classifier",
            "oof_detector": "oof-detector",
            "sc_segmenter": "sc-segmenter",
        }
        model_type = alias_map.get(model_type, model_type)

        if "model_path" not in config_dic:
            raise KeyError(
                f"Required key 'model_path' missing from configuration for {model_type}"
            )

        model_path = config_dic["model_path"]
        if model_type != "pi-classifier":
            if "model_metadata" not in config_dic:
                raise KeyError(
                    f"Required key 'model_metadata' missing from configuration for {model_type}"
                )
            model_configs = read_metadata(config_dic["model_metadata"])

        if model_type == "segmentator":
            model_graph = monai_unet(**model_configs["model_args"])
            self.image_segmentator = Segmentator(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
            self.main_logger.info(f"Loaded model: segmentation (Monai UNet)")
        elif model_type == "cell-classifier":
            model_graph = FlexResNet(**model_configs["model_args"])
            self.object_classifier = CellClassifier(
                model_path, model_graph=model_graph, **config_dic["model_args"]
            )
            self.main_logger.info(f"Loaded model: cell_classifier (FlexResNet)")
        elif model_type == "focus-restorer-fl":
            model_graph = NAFNet(**model_configs["model_args"])
            self.fl_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
            self.main_logger.info(f"Loaded model: fl_focus (NAFNet)")
        elif model_type == "focus-restorer-bf":
            model_graph = NAFNet(**model_configs["model_args"])
            self.bf_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
            self.main_logger.info(f"Loaded model: bf_focus (NAFNet)")
        elif model_type == "pi-classifier":
            self.pi_classifier = PIClassifier(model_path)
            self.main_logger.info(f"Loaded model: pi_classification (scikit-learn)")
        elif model_type == "oof-detector":
            self.oof_detector = OofDetector(
                model_path,
                model_type=model_configs.get("model_type", "rfdetrbase"),
                **config_dic.get("inferer_args", {}),
            )
            self.oof_class_map = config_dic.get("inferer_args", {}).get("class_dict")
            self.main_logger.info(f"Loaded model: oof_detector (RF-DETR)")
        elif model_type == "sc-segmenter":
            self.sc_segmenter = ScSegmenter(
                model_path,
                model_type=model_configs.get("model_type", "rfdetrsegpreview"),
                **config_dic.get("inferer_args", {}),
            )
            self.class_dict = config_dic.get("inferer_args", {}).get("class_dict")
            self.main_logger.info(f"Loaded model: sc_segmenter (RF-DETR Segmenter)")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def load_model_bundle(self, path_to_bundle: str) -> None:
        """
        Load models and configurations from a model bundle.
        The model bundle must be a zip file with the following structure:
        model_bundle.zip
        ├── config.yml          # Main configuration file
        ├── models/             # Directory containing model weights
        │   ├── model_x.pth
        └── metadata/           # Directory containing model metadata
            ├── model_x.json

        Only models required by this pipeline (defined in required_models class attribute)
        will be loaded from the bundle.

        Args:
            path_to_bundle (str): Path to the model bundle zip file.

        Raises:
            FileNotFoundError: If the bundle path does not exist or is not a file.
            ValueError: If the bundle is not a zip file or has an invalid structure.
            AttributeError: If the pipeline does not define required_models.
        """

        if not os.path.isfile(path_to_bundle):
            raise FileNotFoundError(f"Model bundle not found at {path_to_bundle}")
        if not path_to_bundle.endswith(".zip"):
            raise ValueError("Model bundle must be a .zip file")

        # Get pipeline's required models - MANDATORY for selective loading
        required_models = getattr(self, 'required_models', None)
        if not required_models:
            raise AttributeError(
                f"Pipeline '{self.__class__.__name__}' does not have 'required_models' class attribute.\n"
                f"Please define required_models in the pipeline class as a set of model keys.\n"
            )

        self.main_logger.info(
            f"Selective loading enabled for {self.__class__.__name__}. "
            f"Required models: {', '.join(sorted(required_models))}"
        )

        # Define mapping between config keys and internal model types for proper loading
        model_type_mapping = {
            "bf_focus": "focus-restorer-bf",
            "fl_focus": "focus-restorer-fl",
            "segmentation": "segmentator",
            "cell_classifier": "cell-classifier",
            "pi_classification": "pi-classifier",
            "oof_detector": "oof-detector",
            "sc_segmenter": "sc-segmenter",
        }

        try:
            with zipfile.ZipFile(path_to_bundle, "r") as bundle_zip:
                namelist = bundle_zip.namelist()
                # Verify bundle structure has all required components
                required_items = ["config.yml", "models/", "metadata/"]
                for item in required_items:
                    if not any(name.startswith(item) for name in namelist):
                        raise ValueError(
                            f"Invalid model bundle structure: Missing {item}"
                        )

                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_zip.extractall(temp_dir)

                    # Load the main configuration file
                    config_path = os.path.join(temp_dir, "config.yml")
                    with open(config_path, "r") as config_file:
                        config = yaml.safe_load(config_file)

                    # Track loaded and skipped models for summary
                    loaded_models = []
                    skipped_models = []

                    # Process each model in the bundle
                    for model_key, model_config in config.items():
                        if model_key in model_type_mapping:
                            # Skip models not required by this pipeline (ALWAYS selective)
                            if model_key not in required_models:
                                self.main_logger.info(
                                    f"Skipping {model_key} (not required by {self.__class__.__name__})"
                                )
                                skipped_models.append(model_key)
                                continue

                            model_type = model_type_mapping[model_key]

                            # Update paths to point to files in the temporary directory
                            if model_key == "pi_classification":
                                model_config["model_path"] = os.path.join(
                                    temp_dir, model_config["model_path"]
                                )
                            else:
                                model_config["model_path"] = os.path.join(
                                    temp_dir, model_config["model_path"]
                                )
                                model_config["model_metadata"] = os.path.join(
                                    temp_dir, model_config["model_metadata"]
                                )

                            # Load the model
                            self.main_logger.info(f"Loading {model_key}...")
                            self.load_model_fromdict(model_type, model_config)
                            loaded_models.append(model_key)
                        else:
                            self.main_logger.warning(
                                f"Unknown model key in bundle: {model_key}"
                            )

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid or corrupted zip file: {path_to_bundle}")
        except Exception as e:
            self.main_logger.error(f"Error loading model bundle: {e}")
            raise

        # Log summary of loaded models
        self.main_logger.info(
            f"Successfully loaded model bundle: {path_to_bundle}"
        )
        self.main_logger.info(
            f"Models loaded ({len(loaded_models)}): {', '.join(loaded_models)}"
        )
        if skipped_models:
            self.main_logger.info(
                f"Models skipped ({len(skipped_models)}): {', '.join(skipped_models)}"
            )

    def load_tracker(
        self, config_path: str, tracker_override_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load and configure cell tracker from config file or zip bundle.

        Args:
            config_path: Path to config file (.yml/.json) or zip bundle
            tracker_override_args: Optional dict to override tracker parameters
        """

        # Validate file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tracker config not found: {config_path}")

        # Load configuration based on file type
        if config_path.endswith(".zip"):
            config = self._load_tracker_config_from_zip(config_path)
            # Apply override arguments if provided
            if tracker_override_args:
                config = update_config(
                    config, tracker_override_args, logger=self.main_logger
                )

            self.cell_tracker = CellTracker(config_dict=config)
        else:
            # For standalone files, pass config_path
            override_args = tracker_override_args or {}
            self.cell_tracker = CellTracker(
                config_dict=config_path, override_args=override_args
            )

        self.main_logger.info(f"Cell tracker loaded from: {config_path}")

    def _load_tracker_config_from_zip(self, zip_path: str) -> Dict[str, Any]:
        """Load tracker config from zip file."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if "config_tracker.yml" not in zip_ref.namelist():
                raise FileNotFoundError("config_tracker.yml not found in zip root")

            with zip_ref.open("config_tracker.yml") as config_file:
                return yaml.safe_load(config_file)

    def config_image_analysis(
        self,
        reference_channel: int,
        align_frames: bool = False,
        method: str = "basicpy_fl",
    ) -> None:
        """
        Configure the image analysis settings.

        Args:
            reference_channel (int): The reference channel for image analysis.
            align_frames (bool, optional): Whether to align frames. Defaults to False.
            method (str, optional): The method to use for image analysis. Defaults to "basicpy".
        """
        self.reference_channel = reference_channel
        self.align_frames = align_frames
        self.method = method

    def load_config_dict(self, config_dict: Dict) -> None:
        """Configure image analysis settings from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Raises:
            ValueError: If required keys are missing or have invalid types
        """
        required_keys = {"reference_channel": int, "align_frames": bool, "method": str, "focus_correction": bool}

        # Validate required keys and types
        for key, expected_type in required_keys.items():
            if key not in config_dict:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(config_dict[key], expected_type):
                raise ValueError(f"Invalid type for {key}. Expected {expected_type}")

        # Set attributes
        for key, value in config_dict.items():
            setattr(self, key, value)

    def get_files(
        self,
        input_path: str,
        output_folder: str,
        pattern: str = None,
        no_reanalyse: bool = True,
    ) -> List[str]:
        """
        Retrieve a list of files from the specified input path, filtered by pattern and extension.

        Args:
            input_path (str): Path to the directory containing input files.
            output_folder (str): Path to the directory where output files will be saved.
            pattern (str, optional): File name pattern to match. Defaults to None.
            no_reanalyse (bool): If True, skip files that have already been analyzed. Defaults to True.

        Returns:
            List[str]: List of file basenames to be processed.
        """
        worklist_path = self.worklist_path
        if pattern is None:
            pattern = ""
        combined_pattern = f"{pattern}*{self.file_type}"

        # Initialize empty file list
        files_to_process = []

        # Case 1: Using a text file list
        if (
            worklist_path is not None
            and os.path.isfile(worklist_path)
            and worklist_path.endswith(".txt")
        ):
            with open(worklist_path, "r") as file:
                files_to_process = [line.strip() for line in file if line.strip()]
                # Ensure all files exist and match pattern
                files_to_process = [
                    f
                    for f in files_to_process
                    if os.path.exists(f)
                    and fnmatch.fnmatch(os.path.basename(f), combined_pattern)
                ]

        # Case 2: Using input directory
        elif os.path.isdir(input_path):
            files_to_process = glob.glob(os.path.join(input_path, combined_pattern))
        else:
            raise ValueError(
                f"Invalid input: {input_path}. Must be either a directory or a .txt file containing file paths."
            )

        if not files_to_process:
            self.main_logger.warning(
                f"No matching files found with pattern: {combined_pattern}"
            )
            return []

        # Remove already analyzed files if requested
        if no_reanalyse:
            filtered_files = []
            for file_path in files_to_process:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                full_output_path = os.path.join(output_folder, base_name)
                if not all(
                    os.path.exists(full_output_path + ext)
                    for ext in ["_summary.csv", "_fl.csv"]
                ):
                    filtered_files.append(file_path)
                else:
                    self.main_logger.info(
                        f"File {base_name} already analysed. Skipping."
                    )
            files_to_process = filtered_files

        # Return just the basenames for consistency
        return [os.path.basename(f) for f in files_to_process]

    def process_folder(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
        **kwargs,
    ) -> None:
        """
        Process all files with the matching pattern and file extension in the input folder.

        Args:
            files_pattern (str, optional): Glob pattern to match image files. Defaults to None.
            export_labeled_mask (bool): Whether to export labeled mask images. Defaults to False.
            export_aligned_image (bool): Whether to export aligned images. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the analyse_image method.

        Returns:
            None

        Notes:
            - Either files_pattern or file_list must be provided.
            - This method processes files sequentially, unlike process_folder_parallel.
            - The method will analyze each image file using the analyse_image method.
        """
        self.main_logger.info(f"Running hitmictools version {__version__}")
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(f"Output folder: {self.output_path}")
        self.main_logger.info(f"Files pattern: {files_pattern}")
        self.main_logger.info(f"File type: {self.file_type}")
        self.main_logger.info(get_system_info())

        file_list = self.get_files(
            self.input_path,
            self.output_path,
            files_pattern,
            no_reanalyse=True,
        )

        if not file_list:
            self.main_logger.warning("No files to process. Exiting.")
            return

        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        start_time = time.time()
        for idx, name in enumerate(file_list, 1):
            self.main_logger.info(f"Processing file: {name}")
            self.main_logger.info(f"File number {idx} of {len(file_list)}")
            file_i = os.path.join(self.input_path, name)
            file_start_time = time.time()
            try:
                self.analyse_image(
                    file_i,
                    name,
                    export_labeled_mask=export_labeled_mask,
                    export_aligned_image=export_aligned_image,
                    **kwargs,
                )
                file_end_time = time.time()
                file_elapsed_time = file_end_time - file_start_time
                self.main_logger.info(
                    f"Job {name} has finished in time {file_elapsed_time:.2f} seconds"
                )
            except Exception as e:
                self.main_logger.error(f"Error processing file {name}: {str(e)}")

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        self.main_logger.info(
            f"Total processing time for all files: {total_elapsed_time:.2f} seconds"
        )

    def process_folder_parallel(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Process multiple image files in parallel using multiprocessing.

        Args:
            files_pattern (str, optional): Glob pattern to match image files. Defaults to None.
            export_labeled_mask (bool): Whether to export labeled mask images. Defaults to True.
            export_aligned_image (bool): Whether to export aligned images. Defaults to True.
            num_workers (int, optional): Maximum number of worker processes. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the analyse_image method.

        Returns:
            None

        Notes:
            - Either files_pattern or file_list must be provided.
            - If num_workers is None, it defaults to half the number of CPU cores.
            - This method uses multiprocessing to analyze multiple images in parallel.
            - The analyse_image method is expected to handle its own return values.
        """
        file_list = self.get_files(
            self.input_path,
            self.output_path,
            files_pattern,
            no_reanalyse=True,
        )

        if not file_list:
            self.main_logger.warning("No files to process. Exiting.")
            return

        self.main_logger.info(f"Running hitmictools version {__version__}")
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(f"Output folder: {self.output_path}")
        self.main_logger.info(f"Files pattern: {files_pattern}")
        self.main_logger.info(f"File type: {self.file_type}")
        self.main_logger.info(get_system_info())
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        total_cpu_threads = os.cpu_count()
        if num_workers is None or num_workers == 0:
            num_workers = max(1, int(total_cpu_threads // 2))
        self.main_logger.info(f"Total CPU threads: {total_cpu_threads}")
        self.main_logger.info(f"Number of threads used: {num_workers}")
        self.main_logger.info(f"Total files to process: {len(file_list)}")
        start_time = time.time()  # Start timing the entire loop

        try:
            if get_device().type == "cuda":
                # IMPORTANT: spawn required for CUDA; ThreadPoolExecutor would only use
                # threads within individual cores, severely limiting parallelism.
                mp_context = multiprocessing.get_context("spawn")
                self.main_logger.info(
                    "Using spawn context with ProcessPoolExecutor for CUDA backend"
                )
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers, mp_context=mp_context
                )
            elif get_device().type == "mps":
                # IMPORTANT: macOS does not work well with ProcessPoolExecutor (deadlocks,
                # global state loss); ThreadPoolExecutor used instead (torch.compile disabled).
                self.main_logger.info("Using ThreadPoolExecutor for MPS backend")
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                )
            else:
                # IMPORTANT: fork required for CPU; spawn would cause global state loss.
                mp_context = multiprocessing.get_context("fork")
                self.main_logger.info(
                    "Using fork context with ProcessPoolExecutor for CPU backend"
                )
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers, mp_context=mp_context
                )

            with executor:
                futures = {}
                for index, name in enumerate(file_list, start=1):
                    file_i = os.path.join(self.input_path, name)
                    self.main_logger.info(
                        f"Submitting file number {index} of {len(file_list)}"
                    )
                    file_start_time = time.time()
                    future = executor.submit(
                        self.analyse_image,
                        file_i,
                        name,
                        export_labeled_mask,
                        export_aligned_image,
                        **kwargs,
                    )
                    futures[future] = (index, name, file_start_time)

                for future in concurrent.futures.as_completed(futures):
                    index, name, file_start_time = futures[future]
                    try:
                        with managed_resource(future):
                            future.result()
                        file_end_time = time.time()
                        file_elapsed_time = file_end_time - file_start_time
                        self.main_logger.info(
                            f"Job {name} has finished in time {file_elapsed_time:.2f} seconds. ({index}/{len(file_list)})"
                        )
                    except Exception as e:
                        self.main_logger.error(
                            f"Error processing file {index} ({name}): {str(e)}"
                        )
                    finally:
                        gc.collect()
        except Exception as e:
            self.main_logger.error(f"Error in parallel processing: {str(e)}")
        finally:
            end_time = time.time()
            total_elapsed_time = end_time - start_time
            self.main_logger.info(
                f"Total processing time for all files: {total_elapsed_time:.2f} seconds"
            )
            gc.collect()

    @abstractmethod
    def analyse_image(
        self,
        file_path: str,
        file_name: str,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
        **kwargs,
    ) -> None:
        """
        Analyze a single image file.

        This is an abstract method that must be implemented by subclasses.

        Args:
            file_path (str): Full path to the image file.
            file_name (str): Name of the image file.
            export_labeled_mask (bool): Whether to export labeled mask images.
            export_aligned_image (bool): Whether to export aligned images.
            **kwargs: Additional keyword arguments specific to the analysis method.

        Returns:
            None

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass
