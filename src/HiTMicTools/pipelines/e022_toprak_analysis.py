import os
import tifffile
import pandas as pd
import numpy as np
from HiTMicTools.processing_tools import ImagePreprocessor
from HiTMicTools.roi_analyser import (
    RoiAnalyser,
    roi_skewness,
    roi_std_dev,
    rod_shape_coef,
)
from HiTMicTools.workflows import BasePipeline
from HiTMicTools.utils import (
    get_timestamps,
    measure_background_intensity,
    convert_image,
    get_memory_usage
)
from jetraw_tools.image_reader import ImageReader
import psutil


class analysis_e022_sttl(BasePipeline):
    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
        subset_img='test',

    ) -> None:
        """Pipeline analysis for each image."""


        # 1. Read Image:
        movie_name = os.path.splitext(name)[0]
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        align_frames = self.align_frames
        method = self.method

        img_logger.info(f"1 - Reading image, Memory:{get_memory_usage()}")
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        name = os.path.splitext(name)[0]
        pixel_size = metadata.images[0].pixels.physical_size_x
        size_x = metadata.images[0].pixels.size_x
        size_y = metadata.images[0].pixels.size_y
        nSlices=metadata.images[0].pixels.size_z
        nChannels=metadata.images[0].pixels.size_c
        nFrames=metadata.images[0].pixels.size_t

        img=img.reshape(nFrames, nChannels, size_x, size_y)

        if subset_img == 'half':
            n = int(nFrames/2)
            img = img[:n]
        elif subset_img == 'test':
            n = int(nFrames/20)
            img = img[:n]
        nFrames = img.shape[0]

        # 2 Pre-process image
        ip = ImagePreprocessor(img, stack_order="TCXY")
        img=np.zeros((1, 1, 1, 1)) # Remove img to save memory

        # 2.1 Remove background
        img_logger.info(f"2.1 - Preprocessing image, Memory:{get_memory_usage()}")
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
        img_logger.info(f"2.2 - Preprocessing image, Memory:{get_memory_usage()}")
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
            img_logger.info(f"2.3 - Aligning frames in the stack, Memory:{get_memory_usage()}")
            ip.align_image(0, 0, compres_align=.5, crop_image=False, reference="previous")
            img_logger.info(f"2.3 - Alignment completed! Memory:{get_memory_usage()}")
        
        # 2.4 Remove orignal image (not used after background corr) to save mem 
        img_logger.info("Extracting background fluorescence intensity")
        bck_fl = measure_background_intensity(ip.img_original, channel=1)
        ip.img_original=np.zeros((1, 1, 1, 1, 1))

        # 3.1 Segment
        img_logger.info(f"3.1 - Starting segmentation, Memory:{get_memory_usage()}")
        prob_map = self.image_classifier_args.predict(
            ip.img[:, 0, reference_channel, :, :]
        )
        img_logger.info(f"3.1 - Segmentation completed! Memory:{get_memory_usage()}")

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
        img_logger.info(f"3.2 - Extracting ROIs, Memory:{get_memory_usage()}")
        img_analyser = RoiAnalyser(ip.img, prob_map, stack_order=("TSCXY", "TCXY"))
        img_analyser.create_binary_mask()
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found")

        # 4. Calc. measurements
        img_logger.info(f"4 - Starting measurements, Memory:{get_memory_usage()}")
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
        #bck_fl = measure_background_intensity(ip.img_original, channel=1)
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        morpho_measurements = pd.merge(
            morpho_measurements, time_data, on="frame", how="left"
        )
        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info(f"4 - Measurements completed, Memory:{get_memory_usage()}")

        # 4.1 Object classification
        if self.object_classifier is not None:
            img_logger.info(f"4.1 - Running object classification, Memory:{get_memory_usage()}")
            predictions = self.object_classifier.predict(
                morpho_measurements[self.object_classifier.feature_names_in_]
            )
            fl_measurements["object_class"] = predictions
            morpho_measurements["object_class"] = predictions

        # 4.2 PI classification
        if self.pi_classifier is not None:
            img_logger.info(f"4.2 - Running PI classification, Memory:{get_memory_usage()}")
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
            image_8bit = convert_image(ip.img, np.uint8)
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

        img_logger.info(f"Analysis completed for {movie_name}")
        self.remove_logger(img_logger)
        
        return prob_map, ip, metadata