import os
import glob
import time
import fnmatch
import zipfile
import tempfile
import yaml
import logging
from logging.handlers import MemoryHandler
import concurrent.futures
import multiprocessing
from typing import List, Dict, Optional, Any
from HiTMicTools.memlogger import MemoryLogger
from HiTMicTools.model_components.segmentation_model import Segmentator
from HiTMicTools.model_components.cell_classifier import CellClassifier
from HiTMicTools.model_components.focus_restorer import FocusRestorer
from HiTMicTools.utils import get_system_info, read_metadata, get_device

import gc
from contextlib import contextmanager


from HiTMicTools.model_arch.nafnet import NAFNet
from HiTMicTools.model_arch.flexresnet import FlexResNet
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.model_components.pi_classifier import PIClassifier

@contextmanager
def managed_resource(*objects):
    yield objects
    for obj in objects:
        del obj
    gc.collect()


class BasePipeline:
    """
    A class for performing standard analysis on microscopy images.

    Methods:
        setup_logger: Set up a logger for logging the analysis progress.
        remove_logger: Remove logger, useful for concurrent parallel processing.
        load_model: Load a model based on the specified model type.
        config_image_analysis: Configure the image analysis settings.
        get_files: Retrieve a list of files from the specified input path, filtered by pattern and extension.
        process_folder: Process all files with the matching pattern and file extension in the input folder.
        process_folder_parallel: Process multiple image files in parallel using multiprocessing.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        worklist_id: str = "",
        file_type: str = ".nd2",
    ):
        """Initialize the BasePipeline.

        Args:
            input_path (str): Path to the input directory containing the images.
            output_path (str): Path to the output directory for saving the analysis results.
            worklist_id (str, optional): Identifier for the worklist. Defaults to "".
            file_type (str, optional): File extension of the image files. Defaults to '.nd2'.
        """
        last_folder = os.path.basename(os.path.normpath(input_path))
        self.main_logger = self.setup_logger(
            output_path, last_folder, logger_id=worklist_id, print_output=True
        )
        self.input_path = input_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

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
            self.image_classifier = Segmentator(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
        if model_type == "segmentator2":
            model_graph = monai_unet(**model_configs["model_args"])
            self.image_segmentator = Segmentator(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
        elif model_type == "cell-classifier":
            model_graph = FlexResNet(**model_configs["model_args"])
            self.object_classifier = CellClassifier(
                model_path, model_graph=model_graph, **config_dic["model_args"]
            )
        elif model_type == "focus-restorer-fl":
            model_graph = NAFNet(**model_configs["model_args"])
            self.fl_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
        elif model_type == "focus-restorer-bf":
            model_graph = NAFNet(**model_configs["model_args"])
            self.bf_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **config_dic["inferer_args"]
            )
        elif model_type == "pi-classifier":
            self.pi_classifier = PIClassifier(model_path)
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

        Args:
            path_to_bundle (str): Path to the model bundle zip file.

        Raises:
            FileNotFoundError: If the bundle path does not exist or is not a file.
            ValueError: If the bundle is not a zip file or has an invalid structure.
        """

        if not os.path.isfile(path_to_bundle):
            raise FileNotFoundError(f"Model bundle not found at {path_to_bundle}")
        if not path_to_bundle.endswith(".zip"):
            raise ValueError("Model bundle must be a .zip file")

        # Define mapping between config keys and internal model types for proper loading
        model_type_mapping = {
            'bf_focus': 'focus-restorer-bf',
            'fl_focus': 'focus-restorer-fl',
            'segmentation': 'segmentator2',
            'cell_classifier': 'cell-classifier',
            'pi_classification': 'pi-classifier'
        }

        try:
            with zipfile.ZipFile(path_to_bundle, "r") as bundle_zip:
                namelist = bundle_zip.namelist()
                # Verify bundle structure has all required components
                required_items = ["config.yml", "models/", "metadata/"]
                for item in required_items:
                    if not any(name.startswith(item) for name in namelist):
                        raise ValueError(f"Invalid model bundle structure: Missing {item}")

                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_zip.extractall(temp_dir)

                    # Load the main configuration file
                    config_path = os.path.join(temp_dir, 'config.yml')
                    with open(config_path, 'r') as config_file:
                        config = yaml.safe_load(config_file)

                    # Process each model in the bundle
                    for model_key, model_config in config.items():
                        if model_key in model_type_mapping:
                            model_type = model_type_mapping[model_key]
                            
                            # Update paths to point to files in the temporary directory
                            if model_key == 'pi_classification':
                                model_config['model_path'] = os.path.join(temp_dir, model_config['model_path'])
                            else:
                                model_config['model_path'] = os.path.join(temp_dir, model_config['model_path'])
                                model_config['model_metadata'] = os.path.join(temp_dir, model_config['model_metadata'])

                            # Load the model
                            self.load_model_fromdict(model_type, model_config)
                        else:
                            self.main_logger.warning(f"Unknown model key in bundle: {model_key}")

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid or corrupted zip file: {path_to_bundle}")
        except Exception as e:
            self.main_logger.error(f"Error loading model bundle: {e}")
            raise

        self.main_logger.info(f"Successfully validated and loaded model bundle: {path_to_bundle}")


    def load_model(
        self,
        model_type: str,
        model_path: str,
        model_graph: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a model based on the specified model type.

        Args:
            model_type (str): Type of the model to load ('segmentator', 'cell-classifier', 'fl-focus-restorer', 'bf-focus-restorer', 'pi-classifier').
            model_path (str): Path to the model file.
            model_args (str, optional): Graph with the model architecture. Required for 'segmentator' and 'cell-classifier'.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            The loaded model object.

        Raises:
            ValueError: If an invalid model type is provided or required arguments are missing.
        """
        if model_type == "segmentator":
            self.image_classifier = Segmentator(
                model_path, model_graph=model_graph, **kwargs
            )
        if model_type == "segmentator2":
            self.image_segmentator = Segmentator(
                model_path, model_graph=model_graph, **kwargs
            )
        elif model_type == "cell-classifier":
            self.object_classifier = CellClassifier(
                model_path, model_graph=model_graph, **kwargs
            )
        elif model_type == "focus-restorer-fl":
            self.fl_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **kwargs
            )
        elif model_type == "focus-restorer-bf":
            self.bf_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, **kwargs
            )
        elif model_type == "pi-classifier":
            self.pi_classifier = PIClassifier(model_path)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

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
        required_keys = {"reference_channel": int, "align_frames": bool, "method": str}

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
        file_list: str = None,
        pattern: str = None,
        no_reanalyse: bool = True,
    ) -> List[str]:
        """
        Retrieve a list of files from the specified input path, filtered by pattern and extension.

        Args:
            input_path (str): Path to the directory containing input files.
            output_folder (str): Path to the directory where output files will be saved.
            file_list (str, optional): Path to a text file containing a list of input files. Defaults to None.
            pattern (str, optional): File name pattern to match. Defaults to None.
            no_reanalyse (bool): If True, skip files that have already been analyzed. Defaults to True.

        Returns:
            List[str]: List of file names to be processed.
        """
        if pattern is None:
            pattern = ""
        combined_pattern = f"{pattern}*{self.file_type}"
        
        # Initialize empty file list
        files_to_process = []
        
        # Case 1: Using a text file list
        if file_list is not None and os.path.isfile(file_list) and file_list.endswith(".txt"):
            with open(file_list, "r") as file:
                files_to_process = [line.strip() for line in file if line.strip()]
                # Ensure all files exist and match pattern
                files_to_process = [
                    f for f in files_to_process 
                    if os.path.exists(f) and fnmatch.fnmatch(os.path.basename(f), combined_pattern)
                ]
                
        # Case 2: Using input directory
        elif os.path.isdir(input_path):
            files_to_process = glob.glob(os.path.join(input_path, combined_pattern))
        else:
            raise ValueError(
                f"Invalid input: {input_path}. Must be either a directory or a .txt file containing file paths."
            )

        if not files_to_process:
            self.main_logger.warning(f"No matching files found with pattern: {combined_pattern}")
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
                    self.main_logger.info(f"File {base_name} already analysed. Skipping.")
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
            file_list (str, optional): Path to a text file containing image file paths. Defaults to None.
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
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(get_system_info())

        file_list = self.get_files(
            self.input_path,
            self.output_path,
            file_list,
            files_pattern,
            no_reanalyse=True,
        )
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        start_time = time.time()
        for name in file_list:
            self.main_logger.info(f"Processing file: {name}")
            self.main_logger.info(
                f"File number {file_list.index(name)+1} of {len(file_list)}"
            )
            file_i = os.path.join(self.input_path, name)
            file_start_time = time.time()
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
            file_list (str, optional): Path to a text file containing image file paths. Defaults to None.
            export_labeled_mask (bool): Whether to export labeled mask images. Defaults to True.
            export_aligned_image (bool): Whether to export aligned images. Defaults to True.
            num_workers (int, optional): Maximum number of worker processes. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the analyse_image method.

        Returns:
            None

        Notes:
            - Either files_pattern or file_list must be provided.
            - If num_workers is None, it defaults to the number of CPU cores.
            - This method uses multiprocessing to analyze multiple images in parallel.
        """

        file_list = self.get_files(
            self.input_path,
            self.output_path,
            file_list,
            files_pattern,
            no_reanalyse=True,
        )
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(get_system_info())
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        total_cpu_threads = os.cpu_count()
        if num_workers is None or num_workers == 0:
            num_workers = int(total_cpu_threads // 2)
        self.main_logger.info(f"Total CPU threads: {total_cpu_threads}")
        self.main_logger.info(f"Number of threads used: {num_workers}")
        self.main_logger.info(f'Total files to process: {len(file_list)}')
        start_time = time.time()  # Start timing the entire loop

        if get_device().type == 'cuda':
            mp_context = multiprocessing.get_context('spawn')
            self.main_logger.info("Using spawn context with ProcessPoolExecutor for CUDA backend")
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context)
        elif get_device().type == 'mps':
            self.main_logger.info("Using ThreadPoolExecutor for MPS backend")
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        else:
            mp_context = multiprocessing.get_context('fork')
            self.main_logger.info("Using fork context with ProcessPoolExecutor for CPU backend")
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context)

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
                try:
                    with managed_resource(future):
                        result = future.result()
                    index, name, file_start_time = futures[future]
                    file_end_time = time.time()
                    file_elapsed_time = file_end_time - file_start_time
                    self.main_logger.info(
                        f"Job {name} has finished in time {file_elapsed_time:.2f} seconds"
                    )
                    gc.collect()  # Force garbage collection after each file
                except Exception as e:
                    index, name, start_time = futures[future]
                    self.main_logger.error(
                        f"Error processing file {index} ({name}): {e}"
                    )

        end_time = time.time()  # End timing the entire loop
        total_elapsed_time = end_time - start_time
        self.main_logger.info(
            f"Total processing time for all files: {total_elapsed_time:.2f} seconds"
        )
        gc.collect()  # Final garbage collection after all files are processed
