import os
import glob
import fnmatch
import joblib
import logging
from logging.handlers import MemoryHandler
import concurrent.futures
from typing import List, Dict, Optional, Union
from HiTMicTools.model_components.segmentation_model import Segmentator
from HiTMicTools.model_components.cell_classifier import CellClassifier
from HiTMicTools.utils import get_system_info
import gc
from contextlib import contextmanager


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
        worklist_id : str  = "",
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
            output_path, last_folder, logger_id = worklist_id, print_output=True
        )
        self.input_path = input_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path
        self.file_type = file_type

    def setup_logger(
        self, output_path: str, name: str, logger_id : str = "", print_output: bool = False
    ) -> logging.Logger:
        """Set up a logger for logging the analysis progress."""

        # Set up logger file
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

    def load_model(self, 
                  model_type: str, 
                  model_path: str, 
                  model_graph: Optional[str] = None, 
                  **kwargs):
        """
        Load a model based on the specified model type.

        Args:
            model_type (str): Type of the model to load ('segmentator', 'cell-classifier', 'fl-classifier').
            model_path (str): Path to the model file.
            model_args (str, optional): Graph with the model architecture. Required for 'segmentator' and 'cell-classifier'.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            The loaded model object.

        Raises:
            ValueError: If an invalid model type is provided or required arguments are missing.
        """
        if model_type == 'segmentator':
            self.image_classifier = Segmentator(model_path, model_graph, **kwargs)
        
        elif model_type == 'cell-classifier':
            self.object_classifier=CellClassifier(model_path, model_graph, **kwargs)
        
        elif model_type == 'pi-classifier':
            with open(model_path, "rb") as file:
                self.pi_classifier = joblib.load(file)
        
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def config_image_analysis(
        self,
        reference_channel: int,
        align_frames: bool = False,
        method: str = "basicpy",
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

    def get_files(self, input_path: str, output_folder: str, file_list: str = None, pattern: str = None, no_reanalyse: bool = True) -> List[str]:
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

        Notes:
            - If file_list is provided, it takes precedence over input_path.
            - If no_reanalyse is True, files with existing output will be excluded from the returned list.
            - The method supports input as a directory, a text file with file paths, or a list of files.
        """
        if pattern is None:
            pattern = ""
        combined_pattern=f"{pattern}*{self.file_type}"
        if os.path.isdir(input_path) and file_list is None:
            file_list = glob.glob(os.path.join(input_path, combined_pattern))
        elif isinstance(file_list, str) and file_list.endswith(".txt") and os.path.exists(file_list):
            with open(file_list, "r") as file:
                file_list = [line.rstrip() for line in file]
        elif isinstance(file_list, list):
            file_list = [file for file in file_list if fnmatch.fnmatch(file, combined_pattern)]
        else:
            raise ValueError("Invalid input path. It should be a directory, a .txt file, or a list of files.")
        
        # Remove files that have already been analysed
        if no_reanalyse:
            for file_i in file_list:
                full_path=os.path.join(output_folder, os.path.splitext(file_i)[0])
                if all(
                    os.path.exists(full_path + ext)
                    for ext in ["_summary.csv", "_fl.csv"]
                ):
                    file_list.remove(file_i)
                    self.main_logger.info(f"File {file_i} already analysed. Skipping.")
        
        file_list = [os.path.basename(file_i) for file_i in file_list]
        return file_list

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

        file_list=self.get_files(self.input_path, self.output_path, file_list, files_pattern, no_reanalyse=True)
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        for name in file_list:
            self.main_logger.info(f"Processing file: {name}")
            self.main_logger.info(
                f"File number {file_list.index(name)+1} of {len(file_list)}"
            )
            file_i = os.path.join(self.input_path, name)
            start_time = time.time()
            self.analyse_image(
                file_i,
                name,
                export_labeled_mask=export_labeled_mask,
                export_aligned_image=export_aligned_image,
                **kwargs,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.main_logger.info(f"Job {name} has finished in time {elapsed_time:.2f} seconds")

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

        file_list=self.get_files(self.input_path, self.output_path, file_list, files_pattern, no_reanalyse=True)
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
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for index, name in enumerate(file_list, start=1):
                file_i = os.path.join(self.input_path, name)
                self.main_logger.info(f"Submitting file number {index} of {len(file_list)}")
                start_time = time.time()
                future = executor.submit(self.analyse_image, file_i, name, export_labeled_mask, export_aligned_image, **kwargs)
                futures[future] = (index, name, start_time)

            for future in concurrent.futures.as_completed(futures):
                try:
                    with managed_resource(future):
                        result = future.result()
                    index, name, start_time = futures[future]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.main_logger.info(f"Job {name} has finished in time {elapsed_time:.2f} seconds")
                    gc.collect()  # Force garbage collection after each file
                except Exception as e:
                    index, name, start_time = futures[future]
                    self.main_logger.error(f"Error processing file {index} ({name}): {e}")

        gc.collect()  # Final garbage collection after all files are processed