import os
import gc
import tifffile
from typing import Optional, List, Union, Dict
import torch
import pandas as pd
import numpy as np

# Local imports
from HiTMicTools.resource_management.memlogger import MemoryLogger
from HiTMicTools.resource_management.sysutils import (
    empty_gpu_cache,
    get_device,
    )
from HiTMicTools.resource_management.reserveresource import ReserveResource
from HiTMicTools.workflows import BasePipeline
from HiTMicTools.img_processing.img_processor import ImagePreprocessor
from HiTMicTools.img_processing.array_ops import convert_image
from HiTMicTools.img_processing.img_ops import measure_background_intensity
from HiTMicTools.img_processing.mask_ops import map_predictions_to_labels
from HiTMicTools.utils import get_timestamps, remove_file_extension
from HiTMicTools.roi_analyser import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev

# TODO: Currently, I can use the cupy based ROI analyser, but performance is lagging.
# I will start working with the CPU-based ROI analyser and slowly move to the GPU-based.
# if get_device() == torch.device("cuda"):
#    from HiTMicTools.roi_analyser_gpu import RoiAnalyser, roi_skewness, roi_std_dev
#    import GPUtil
#    print('using CUDA based ROI analyser')
#
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev
#    print('using CPU based ROI analyser')
#
# else:
#    print('using CPU based ROI analyser')
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev


from jetraw_tools.image_reader import ImageReader
import psutil
import time
import random


class ASCT_zaslavier(BasePipeline):
    """
    Pipeline for automated single-cell tracking with focus restoration.

    This pipeline processes microscopy images to:
    1. Restore focus in both brightfield and fluorescence channels
    2. Segment and classify cells in the images
    3. Track cells across time frames
    4. Analyze fluorescence intensity and other cellular properties

    The pipeline is designed for time-lapse microscopy data with multiple channels,
    particularly for experiments tracking PI (propidium iodide) uptake in cells.

    Attributes:
        reference_channel (int): Index of the brightfield/reference channel
        pi_channel (int): Index of the fluorescence/PI channel
        align_frames (bool): Whether to align frames in time series
        method (str): Background correction method ('standard', 'basicpy', or 'basicpy_fl')
        image_segmentator: Model for cell segmentation
        object_classifier: Model for classifying segmented objects
        bf_focus_restorer: Model for restoring focus in brightfield images
        fl_focus_restorer: Model for restoring focus in fluorescence images
        pi_classifier: Model for classifying PI positive/negative cells
    """

    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
    ) -> None:
        """Pipeline analysis for each image."""

        # 1. Read Image:
        device = get_device()
        is_cuda = device == torch.device("cuda")
        movie_name = remove_file_extension(name)
        name = movie_name
        # Desync analysis to avoid RAM/VRAM issues
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        pi_channel = self.pi_channel
        gfp_channel = getattr(self, "gfp_channel", None)
        align_frames = self.align_frames
        tracking = self.tracking
        method = self.method

        img_logger.info("1 - Reading image", show_memory=True)
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        pixel_size = metadata.images[0].pixels.physical_size_x
        size_x = metadata.images[0].pixels.size_x
        size_y = metadata.images[0].pixels.size_y
        nSlices = metadata.images[0].pixels.size_z
        nChannels = metadata.images[0].pixels.size_c
        nFrames = metadata.images[0].pixels.size_t
        # 2 Pre-process image --------------------------------------------
        img_logger.info(
            f"Image shape: {img.shape}, pixel size: {pixel_size} µm. Reshaped to (frames={nFrames}, channels={nChannels}, slices={nSlices}, x={size_x}, y={size_y})"
        )
        img = img.reshape(nFrames, nChannels, size_x, size_y)

        ip = ImagePreprocessor(img, stack_order="TCXY")
        img = np.zeros((1, 1, 1, 1))  # Remove img to save memory

        # 2.1 Remove background
        img_logger.info("2.1 - Preprocessing image", show_memory=True)
        img_logger.info(f"Preprocessed image shape: {ip.img.shape}")

        # 2.3 Align frames if required
        if align_frames:
            img_logger.info("2.1 - Aligning frames in the stack", show_memory=True)
            ip.align_image(
                ref_channel=0, ref_slice=-1, crop_image=True, reference_type="previous"
            )
            img_logger.info("2.1 - Frame alignment completed", show_memory=True)
        # Update size x and size y after alignment and maybe crop
        size_x, size_y = ip.img.shape[-2], ip.img.shape[-1]
        img_logger.info("2.1 - Detecting and fixing border wells")
        ip.detect_fix_well(nchannels=0, nslices=0, nframes=range(nFrames))
        img_logger.info(
            f"Reference channel intensity before background removal:\n{self.check_px_values(ip, reference_channel, round=3)}"
        )

        self.clear_background(
            ip,
            channel=reference_channel,
            nFrames=range(nFrames),
            method=method,
            pixel_size=pixel_size,
        )
        self.clear_background(
            ip, channel=pi_channel, nFrames=range(nFrames), method=method
        )
        self.clear_background(
            ip, channel=gfp_channel, nFrames=range(nFrames), method=method
        )

        # 2.2 Focus restoration in the reference channel
        img_logger.info(
            "2.2 - Restoring focus in the reference channel", show_memory=True
        )
        img_logger.info(
            f"Reference channel intensity before focus restoration:\n{self.check_px_values(ip, reference_channel, round=3)}"
        )
        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            ip.img[:, 0, reference_channel] = self.bf_focus_restorer.predict(
                ip.img[:, 0, reference_channel],
                rescale=False,
                batch_size=1,
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )
        img_logger.info("2.2 - Restoring focus in the PI channel", show_memory=True)
        img_logger.info(
            f"PI channel intensity before focus restoration:\n{self.check_px_values(ip, pi_channel, round=3)}"
        )
        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            ip.img[:, 0, pi_channel] = self.fl_focus_restorer.predict(
                ip.img[:, 0, pi_channel],
                batch_size=1,
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
                padding_mode="reflect",
            )
        
        if gfp_channel is not None:
            img_logger.info("2.2 - Restoring focus in the GFP channel", show_memory=True)
            with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
                ip.img[:, 0, gfp_channel] = self.fl_focus_restorer.predict(
                    ip.img[:, 0, gfp_channel],
                    batch_size=1,
                    buffer_steps=4,
                    buffer_dim=-1,
                    sw_batch_size=1,
                    padding_mode="reflect",
                )
        img_logger.info(
            f"Reference channel intensity after focus restoration:\n{self.check_px_values(ip, reference_channel, round=3)}"
        )
        img_logger.info(
            f"PI channel intensity after focus restoration:\n{self.check_px_values(ip, pi_channel, round=3)}"
        )

        # 2.4 Remove original image (not used after background corr) to save mem
        ip.img_original = np.zeros((1, 1, 1, 1, 1))

        # 3.1 Segment Image --------------------------------------------
        img_logger.info("3.1 - Image segmentation", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 4.0, logger=img_logger, timeout=120):
            prob_map = self.image_segmentator.predict(
                ip.img[:, 0, reference_channel, :, :],
                buffer_steps=4,
                buffer_dim=-1,
                sw_batch_size=1,
            )
        img_logger.info("3.1 - Segmentation completed", show_memory=True, cuda=is_cuda)

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
        img_logger.info("3.2 - Extracting ROIs", show_memory=True)
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))

        # Remove image-processor to release space
        del ip
        img_analyser.create_binary_mask()
        img_analyser.clean_binmask(min_pixel_size=20)
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found in segmentation")

        # 3.3 Classify ROIs
        img_logger.info("3.2 - Classifying ROIs", show_memory=True, cuda=is_cuda)
        with ReserveResource(device, 12.0, logger=img_logger, timeout=240):
            object_classes, labels = self.batch_classify_rois(img_analyser, batch_size=4)
        img_logger.info(
            "3.2 - GPU memory status after classification",
            show_memory=True,
            cuda=is_cuda,
        )

        # 4.1 Calc. measurements --------------------------------------------
        img_logger.info("4 - Starting measurements", show_memory=True)
        img_logger.info("4.1 - Extracting background fluorescence intensity")
        bck_fl = measure_background_intensity(
            img_analyser.img, img_analyser.labeled_mask, target_channel=pi_channel
        )

        fl_prop = [
            "label",
            "centroid",
            "max_intensity",
            "min_intensity",
            "mean_intensity",
            "area",
            "major_axis_length",
            "minor_axis_length",
            "solidity",
            "orientation",
        ]
        img_logger.info("4.2 - Extracting fluorescence measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=pi_channel,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev),
        )
        fl_measurements["object_class"] = object_classes

        # If measure_gfp is set, measure mean_intensity from gfp_channel and merge
        if hasattr(self, "measure_gfp"):
            gfp_channel = getattr(self, "gfp_channel", None)
            if gfp_channel is not None:
                # Measure GFP background intensity first
                img_logger.info("4.2b - Extracting GFP background intensity")
                gfp_bck = measure_background_intensity(
                    img_analyser.img,
                    img_analyser.labeled_mask,
                    target_channel=gfp_channel,
                )
                gfp_bck = gfp_bck.rename(columns={"background": "gfp_background"})

                # Measure GFP mean intensity for ROIs
                img_logger.info("4.2c - Extracting GFP mean intensity")
                gfp_measurements = img_analyser.get_roi_measurements(
                    target_channel=gfp_channel,
                    properties=["label", "mean_intensity"],
                )
                # Rename mean_intensity to mean_intensity_gfp to avoid overlap
                gfp_measurements = gfp_measurements.rename(
                    columns={"mean_intensity": "mean_intensity_gfp"}
                )

                # Merge GFP background with GFP measurements on 'frame'
                gfp_merged = pd.merge(gfp_measurements, gfp_bck, on="frame", how="left")

                # Merge the GFP results to the main fluorescence measurements on 'label'
                fl_measurements = pd.merge(
                    fl_measurements, gfp_merged, on="label", how="left"
                )
            else:
                img_logger.warning("measure_gfp is True but gfp_channel is not set.")

        img_logger.info("4.3 - Extracting time metadata")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        fl_measurements[
            ["rel_max_intensity", "rel_min_intensity", "rel_mean_intensity"]
        ] = fl_measurements[["max_intensity", "min_intensity", "mean_intensity"]].div(
            fl_measurements["background"], axis=0
        )

        # 4.4 Object tracking (if enabled)
        if self.tracking and self.cell_tracker is not None:
            img_logger.info("4.4 - Running object tracking")
            track_features = fl_prop[5:10]
            self.cell_tracker.set_features(track_features)
            try:
                fl_measurements = self.cell_tracker.track_objects(
                    fl_measurements, volume_bounds=(size_x, size_y), logger=img_logger
                )
                img_logger.info("4.4 - Object tracking completed successfully")
            except Exception as e:
                img_logger.error(f"Object tracking failed: {e}")
                # Continue without tracking

        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info("4 - Measurements completed", show_memory=True)

        # 4.1 PI classification
        if self.pi_classifier is not None:
            img_logger.info("4.4 - Running PI classification", show_memory=True)
            predictions = self.pi_classifier.predict(
                fl_measurements[self.pi_classifier.feature_names_in_]
            )
            fl_measurements["pi_class"] = predictions
            fl_measurements["file"] = name

            # Generate summary data using the dedicated method
            d_summary = self.generate_data_summary(
                fl_measurements,
                [
                    "file",
                    "frame",
                    "channel",
                    "date_time",
                    "timestep",
                    "abslag_in_s",
                    "object_class",
                ],
                img_logger,
            )
        else:
            d_summary = pd.DataFrame()

        # 5. Export data --------------------------------------------
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing output data to {export_path}")

        fl_measurements.to_csv(export_path + "_fl.csv")
        d_summary.to_csv(export_path + "_summary.csv")

        if export_labeled_mask:
            # Create mapping for object classes
            class_to_id = {
                "single-cell": 0,
                "clump": 1,
                "noise": 2,
                "off-focus": 3,
                "joint-cell": 4,
            }

            # Map object classes to the labeled mask
            object_class_mask = map_predictions_to_labels(
                img_analyser.labeled_mask[:, 0, 0],
                object_classes,
                labels,
                value_map={
                    class_name: class_id + 1
                    for class_name, class_id in class_to_id.items()
                },
            )

            # If PI classifier was used, create a second channel for PI classification
            if self.pi_classifier is not None:
                # Map PI classes to the labeled mask
                pi_class_mask = map_predictions_to_labels(
                    img_analyser.labeled_mask[:, 0, 0],
                    fl_measurements["pi_class"].tolist(),
                    fl_measurements["label"].tolist(),
                    value_map={"piPOS": 1, "piNEG": 2},
                )

                # Stack the two channels: object class and PI class
                combined_mask = np.stack([object_class_mask, pi_class_mask], axis=1)
                labs_8bit = combined_mask.astype(np.uint8)
                axes = "TCYX"
                log_msg = (
                    "Exported labeled mask with object and PI classification channels"
                )
            else:
                # If no PI classifier, just save the object classification channel
                labs_8bit = object_class_mask.astype(np.uint8)
                axes = "TYX"
                log_msg = "Exported labeled mask with object classification channel"

            # Save the labeled mask with appropriate metadata
            tifffile.imwrite(
                export_path + "_labels.tiff",
                labs_8bit,
                imagej=True,
                metadata={"axes": axes},
            )
            img_logger.info(log_msg)
        if export_aligned_image:
            image_8bit = convert_image(img_analyser.img, np.uint8)
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

        img_logger.info(f"Analysis completed for {movie_name}", show_memory=True)
        del prob_map, img, fl_measurements, d_summary, img_analyser
        gc.collect()
        empty_gpu_cache(device)
        img_logger.info("Garbage collection completed", show_memory=True)

        self.remove_logger(img_logger)

        return name

    def batch_classify_rois(self, img_analyser, batch_size=5):
        labeled_mask = img_analyser.labeled_mask[:, 0, 0]  # .get()
        img = img_analyser.img[:, 0, 0]  # .get()

        n_frames = labeled_mask.shape[0]
        all_object_classes = []
        all_labels = []

        for start_frame in range(0, n_frames, batch_size):
            end_frame = min(start_frame + batch_size, n_frames)

            # Extract batch of frames
            batch_labeled_mask = labeled_mask[start_frame:end_frame]
            batch_img = img[start_frame:end_frame]

            # Classify the batch
            batch_classes, batch_labels = self.object_classifier.classify_rois(
                batch_labeled_mask, batch_img
            )

            all_object_classes.extend(batch_classes)
            all_labels.extend(batch_labels)

        return all_object_classes, all_labels

    def clear_background(
        self,
        ip: ImagePreprocessor,
        channel: int,
        nFrames: range,
        method: str,
        pixel_size: Optional[float] = None,
    ) -> None:
        """Remove background from images using specified method.

        Args:
        ip: Image preprocessor object
        channel: Channel to process
        nFrames: Range of frames to process
        method: Background removal method ('standard', 'basicpy', or 'basicpy_fl')
        pixel_size: Physical pixel size in microns
        """
        # If using the basicpy_fl in config, reference channel is still transform with DoG
        pi_channel = getattr(self, "pi_channel", None)
        gfp_channel = getattr(self, "gfp_channel", None)
        if method == "basicpy_fl" and channel == self.reference_channel:
            method = "standard"
        elif method == "basicpy_fl" and channel in (pi_channel, gfp_channel):
            method = "basicpy"

        methods = {
            "standard": [
                {
                    "nframes": nFrames,
                    "nchannels": channel,
                    "nslices": 0,
                    "sigma_r": 20,
                    "method": "divide",
                }
            ],
            "basicpy": [
                {
                    "nframes": nFrames,
                    "nchannels": channel,
                    "nslices": 0,
                    "method": "basicpy",
                    "smoothness_flatfield": 5,
                    "smoothness_darkfield": 5,
                    "get_darkfield": False,
                    "sort_intensity": False,
                    "fitting_mode": "approximate",
                }
            ],
        }

        if method not in methods:
            raise ValueError(f"Invalid method: {method}")

        for params in methods[method]:
            if method == "basicpy":
                ip.clear_image_background(**params)
            else:
                ip.clear_image_background(**params, unit="um", pixel_size=pixel_size)

    def generate_data_summary(
        self,
        fl_measurements: pd.DataFrame,
        by_list: List[str],
        img_logger: MemoryLogger,
    ) -> pd.DataFrame:
        """
        Generate a summary DataFrame from fluorescence measurements with PI classification.

        This method aggregates the fluorescence measurements by file, frame, channel,
        timestamp information, and object class to create a summary of PI-positive and
        PI-negative cell counts and areas.

        Args:
            fl_measurements: DataFrame containing fluorescence measurements with 'pi_class' column.
                Must include columns: 'file', 'frame', 'channel', 'date_time', 'timestep',
                'abslag_in_s', 'object_class', 'label', 'area', and 'pi_class'.
            img_logger: Logger instance for recording progress and errors.

        Returns:
            pd.DataFrame: A summary DataFrame with aggregated counts and areas, or an empty
                DataFrame if an error occurs during the groupby operation.

        Notes:
            The summary includes the following aggregated metrics:
            - total_count: Total number of objects per group
            - pi_class_neg: Count of PI-negative objects
            - pi_class_pos: Count of PI-positive objects
            - area_pineg: Total area of PI-negative objects
            - area_pipos: Total area of PI-positive objects
            - area_total: Total area of all objects
        """
        try:
            img_logger.info(f"Group data by {by_list}")
            d_summary = (
                fl_measurements.groupby(by_list)
                .agg(
                    total_count=("label", "count"),
                    pi_class_neg=("pi_class", lambda x: (x == "piNEG").sum()),
                    pi_class_pos=("pi_class", lambda x: (x == "piPOS").sum()),
                    area_pineg=(
                        "area",
                        lambda x: x[
                            fl_measurements.loc[x.index, "pi_class"] == "piNEG"
                        ].sum(),
                    ),
                    area_pipos=(
                        "area",
                        lambda x: x[
                            fl_measurements.loc[x.index, "pi_class"] == "piPOS"
                        ].sum(),
                    ),
                    area_total=("area", "sum"),
                )
                .reset_index()
            )

            img_logger.info(
                f"Groupby operation completed successfully. Shape of d_summary: {d_summary.shape}"
            )
        except Exception as e:
            img_logger.error(f"Error during groupby operation: {str(e)}")
            img_logger.error(f"Columns in fl_measurements: {fl_measurements.columns}")
            img_logger.error(
                f"Unique values in 'pi_class': {fl_measurements['pi_class'].unique()}"
            )
            d_summary = pd.DataFrame()

        img_logger.info("d_summary created successfully", show_memory=True)

        return d_summary

    @staticmethod
    def check_px_values(ip, channel: int, round: int = None) -> np.ndarray:
        """Calculate mean pixel intensity across frames for a given channel."""
        means = np.mean(ip.img[:, 0, channel], axis=(1, 2))
        return np.round(means, round) if round is not None else means
