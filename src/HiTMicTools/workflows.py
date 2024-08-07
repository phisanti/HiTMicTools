import os
import glob
import fnmatch
import joblib
import logging
from logging.handlers import MemoryHandler
import concurrent.futures
from typing import List, Dict, Optional, Union
from HiTMicTools.segmentation_model import Segmentator
from HiTMicTools.utils import get_system_info

class BasePipeline:
    """
    A class for performing standard analysis on microscopy images.

    Args:
        input_path (str): Path to the input directory containing the images.
        output_path (str): Path to the output directory for saving the analysis results.
        image_classifier_args (Dict[str, Union[str, int, float, bool]]): Arguments for the image segmentation model.
        object_classifier (str): Path to the classifier for object classification.
        pi_classifier (str): Path to the classifier for PiPOS/PiNEG (Propidium Iodide) classification.
        file_type (str, optional): File extension of the image files. Defaults to '.nd2'.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        image_classifier_args: Dict[str, Union[str, int, float, bool]],
        object_classifier: str,
        pi_classifier: str,
        worklist_id : str  = "",
        file_type: str = ".nd2",
    ):
        last_folder = os.path.basename(os.path.normpath(input_path))
        self.main_logger = self.setup_logger(
            output_path, last_folder, logger_id = worklist_id, print_output=True
        )
        self.input_path = input_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path
        self.file_type = file_type
        self.image_classifier_args = self.add_segmentator(image_classifier_args)
        self.object_classifier = self.add_classifier(object_classifier)
        self.pi_classifier = self.add_classifier(pi_classifier)

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

    @staticmethod
    def add_segmentator(
        classifier_args: Dict[str, Union[str, int, float, bool]],
    ) -> Segmentator:
        """Attach segmentator class for pixel classification."""
        return Segmentator(**classifier_args)

    @staticmethod
    def add_classifier(model_path: Optional[str]):
        """Load a classifier model from a file."""
        if model_path is None:
            classifier = None
        else:
            with open(model_path, "rb") as file:
                classifier = joblib.load(file)

        return classifier

    def config_image_analysis(
        self,
        reference_channel: int,
        align_frames: bool = False,
        method: str = "basicpy",
    ) -> None:
        """Configure the image analysis settings."""
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
                    for ext in ["_morpho.csv", "_fl.csv"]
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

            self.analyse_image(
                file_i,
                name,
                export_labeled_mask=export_labeled_mask,
                export_aligned_image=export_aligned_image,
                **kwargs,
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for index, name in enumerate(file_list, start=1):
                file_i = os.path.join(self.input_path, name)
                self.main_logger.info(
                    f"Submitting file number {index} of {len(file_list)}"
                )
                future = executor.submit(
                    self.analyse_image,
                    file_i,
                    name,
                    export_labeled_mask,
                    export_aligned_image,
                    **kwargs,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.main_logger.error(f"Error processing file: {e}")