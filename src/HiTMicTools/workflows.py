import os
import glob
import fnmatch
import tifffile
import joblib
import pandas as pd
import numpy as np
import logging
import concurrent.futures
from typing import List, Dict, Optional, Union
from HiTMicTools.utils import (
    get_timestamps,
    measure_background_intensity,
    convert_image,
)
from HiTMicTools.segmentation_model import Segmentator
from HiTMicTools.processing_tools import ImagePreprocessor
from HiTMicTools.roi_analyser import (
    RoiAnalyser,
    roi_skewness,
    roi_std_dev,
    rod_shape_coef,
)
from jetraw_tools.image_reader import ImageReader


class StandardAnalysis:
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
        file_type: str = ".nd2",
    ):
        last_folder = os.path.basename(os.path.normpath(input_path))
        self.main_logger = self.setup_logger(
            output_path, last_folder, print_output=True
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
        self, output_path: str, name: str, print_output: bool = False
    ) -> logging.Logger:
        """Set up a logger for logging the analysis progress."""

        # Set up logger file
        last_folder = os.path.basename(os.path.normpath(name))
        log_file = os.path.join(output_path, f"{name}_analysis.log")
        logger_name = f"{output_path}_{name}"  # Use a unique identifier for each instance important for parallelisation
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
        If the csv are already present, files can be filtered out. 
        Returns the list of file to be processed.
        """
        if pattern is None:
            pattern = ""
        combined_pattern=f"{pattern}*{self.file_type}"
        if os.path.isdir(input_path) and file_list is None:
            file_list = glob.glob(os.path.join(input_path, combined_pattern))
        elif isinstance(file_list, str) and file_list.endswith(".txt") and os.path.exists(file_list):
            with open(file_list, "r") as file:
                file_list = [line.rstrip() for line in file if fnmatch.fnmatch(line.rstrip(), combined_pattern)]
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
        
        return [os.path.basename(file_i) for file_i in file_list]

    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
    ) -> None:
        """Pipeline analysis for each image."""

        # 1. Read Image:
        movie_name = os.path.splitext(name)[0]
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        align_frames = self.align_frames
        method = self.method

        img_logger.info(f"1 - Reading image")
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        name = os.path.splitext(name)[0]
        pixel_size = metadata.images[0].pixels.physical_size_x
        nFrames = img.shape[0]

        # 2 Pre-process image
        ip = ImagePreprocessor(img, stack_order="TCXY")

        # 2.1 Remove background
        img_logger.info(f"2.1 - Preprocessing image")
        img_logger.info(f"Image shape {ip.img.shape}")
        mean_intensity_0 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity before clear background:\n{np.round(mean_intensity_0, 3)}"
        )

        if method == "standard":
            ip.clear_image_background(
                range(nFrames),
                0,
                0,
                sigma_r=10,
                method="divide",
                unit="um",
                pixel_size=pixel_size,
            )
        elif method == "local_background_fl":
            ip.clear_image_background(
                range(nFrames),
                0,
                nchannels=0,
                sigma_r=50,
                method="divide",
                unit="um",
                pixel_size=pixel_size,
            )
            ip.clear_image_background(
                range(nFrames),
                0,
                nchannels=1,
                sigma_r=50,
                method="subtract",
                unit="um",
                pixel_size=pixel_size,
            )
        elif method == "basicpy_fl":
            ip.clear_image_background(
                range(nFrames),
                0,
                nchannels=0,
                sigma_r=20,
                method="divide",
                unit="um",
                pixel_size=pixel_size,
            )
            ip.clear_image_background(
                range(nFrames),
                0,
                1,
                method="basicpy",
                get_darkfield=False,
                smoothness_flatfield=3,
                sort_intensity=False,
                fitting_mode="approximate",
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        # 2.2 Normalize image intensity in the reference channel
        img_logger.info(f"2.2 - Preprocessing image")
        mean_intensity_1 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity before normalization:\n{np.round(mean_intensity_1, 3)}"
        )
        ip.norm_eq_hist(range(nFrames), 0, nchannels=0)
        mean_intensity_2 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"2.2 - Intensity after normalization:\n{np.round(mean_intensity_2, 3)}"
        )

        # 2.3 Align frames if required
        if align_frames:
            img_logger.info(f"2.3 - Aligning frames in the stack")
            ip.align_image(0, 0, crop_image=False, reference="previous")
            img_logger.info(f"2.3 - Alignment completed!")

        # 3.1 Segment
        img_logger.info(f"3.1 - Starting segmentation")
        prob_map = self.image_classifier_args.predict(
            ip.img[:, 0, reference_channel, :, :]
        )
        img_logger.info(f"3.1 - Segmentation completed!")

        # Get ROIs
        if prob_map.ndim > 3 and prob_map.shape[1] > 1:
            prob_map = np.max(prob_map, axis=1, keepdims=True)
        elif prob_map.ndim == 3:
            prob_map = np.expand_dims(prob_map, axis=1)
        elif prob_map.ndim == 2:
            prob_map = np.expand_dims(prob_map, axis=(0, 1))
        else:
            pass

        # 3.2 Get ROIs
        img_logger.info(f"3.2 - Extracting ROIs")
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))
        img_analyser.create_binary_mask()
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found")

        # 4. Calc. measurements
        img_logger.info(f"4 - Starting measurements")
        morphology_prop = [
            "label",
            "centroid",
            "max_intensity",
            "min_intensity",
            "mean_intensity",
            "eccentricity",
            "solidity",
            "area",
            "perimeter",
            "feret_diameter_max",
            "coords",
        ]
        fl_prop = [
            "label",
            "centroid",
            "max_intensity",
            "min_intensity",
            "mean_intensity",
        ]
        img_logger.info("Extracting morphological measurements")
        morpho_measurements = img_analyser.get_roi_measurements(
            target_channel=0,
            properties=morphology_prop,
            extra_properties=(roi_skewness, roi_std_dev, rod_shape_coef),
            asses_focus=True,
        )
        img_logger.info("Extracting fluorescent measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=1,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev),
            asses_focus=False,
        )
        img_logger.info("Extracting time data")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        img_logger.info("Extracting background fluorescence intensity")
        bck_fl = measure_background_intensity(ip.img_original, channel=1)
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        morpho_measurements = pd.merge(
            morpho_measurements, time_data, on="frame", how="left"
        )
        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info(f"4 - Measurements completed")

        # 4.1 Object classification
        if self.object_classifier is not None:
            img_logger.info(f"4.1 - Running object classification")
            predictions = self.object_classifier.predict(
                morpho_measurements[self.object_classifier.feature_names_in_]
            )
            fl_measurements["object_class"] = predictions
            morpho_measurements["object_class"] = predictions

        # 4.2 PI classification
        if self.pi_classifier is not None:
            img_logger.info(f"4.2 - Running PI classification")
            predictions = self.pi_classifier.predict(
                fl_measurements[self.pi_classifier.feature_names_in_]
            )
            fl_measurements["pi_class"] = predictions

        # 5. Export data
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing data to {export_path}")
        morpho_measurements.to_csv(export_path + "_morpho.csv")
        fl_measurements.to_csv(export_path + "_fl.csv")

        if export_labeled_mask:
            pmap_8bit = convert_image(img_analyser.proba_map, np.uint8)
            tifffile.imwrite(export_path + "_labels.tiff", pmap_8bit)
        if export_aligned_image:
            tifffile.imwrite(export_path + "_transformed.tiff", ip.img, imagej=True)

        img_logger.info(f"Analysis completed for {movie_name}")
        self.remove_logger(img_logger)
        pass

    def process_folder(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
    ) -> None:
        """Process all files with the matching pattern and file extension in the input folder."""

        self.main_logger.info(f"Processing folder: {self.input_path}")
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
            )

    def process_folder_parallel(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        """Process all files in the input folder using parallel processing."""
        file_list=self.get_files(self.input_path, self.output_path, file_list, files_pattern, no_reanalyse=True)
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        total_cpu_threads = os.cpu_count()
        if max_workers is None:
            max_workers = int(total_cpu_threads // 2)
        self.main_logger.info(f"Total CPU threads: {total_cpu_threads}")
        self.main_logger.info(f"Number of threads used: {max_workers}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.main_logger.error(f"Error processing file: {e}")