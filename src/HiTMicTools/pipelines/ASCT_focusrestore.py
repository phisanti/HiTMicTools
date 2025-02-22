import os
import gc
import tifffile
import torch
import pandas as pd
import numpy as np
from HiTMicTools.processing_tools import ImagePreprocessor
from HiTMicTools.workflows import BasePipeline
from HiTMicTools.utils import (
    get_timestamps,
    measure_background_intensity,
    convert_image,
    get_memory_usage,
    remove_file_extension,
    get_device,
    empty_gpu_cache,
)
from HiTMicTools.roi_analyser import RoiAnalyser
from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev

# TODO: Currently, I can use the cupy based ROI analyser, but performance is lagging.
# I will start working with the CPU-based ROI analyser and slowly move to the GPU-based. 
#if get_device() == torch.device("cuda"):
#    from HiTMicTools.roi_analyser_gpu import RoiAnalyser, roi_skewness, roi_std_dev
#    import GPUtil
#    print('using CUDA based ROI analyser')
#
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev
#    print('using CPU based ROI analyser')
#
#else:
#    print('using CPU based ROI analyser')
#    from HiTMicTools.roi_analyser import RoiAnalyser
#    from HiTMicTools.data_analysis.analysis_tools import roi_skewness, roi_std_dev


from jetraw_tools.image_reader import ImageReader
import psutil


class ASCT_focusRestoration(BasePipeline):
    def analyse_image(
        self,
        file_i: str,
        name: str,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True
    ) -> None:
        """Pipeline analysis for each image."""


        # 1. Read Image:
        is_cuda = get_device() == torch.device("cuda")
        if is_cuda:
            import GPUtil
            gpu=GPUtil.getGPUs()[0]
        movie_name = remove_file_extension(name)
        name = movie_name
        img_logger = self.setup_logger(self.output_path, movie_name)
        img_logger.info(f"Start analysis for {movie_name}")
        reference_channel = self.reference_channel
        pi_channel = self.pi_channel
        align_frames = self.align_frames
        method = self.method

        img_logger.info(f"1 - Reading image, Memory:{get_memory_usage()}")
        image_reader = ImageReader(file_i, self.file_type)
        img, metadata = image_reader.read_image()
        pixel_size = metadata.images[0].pixels.physical_size_x
        size_x = metadata.images[0].pixels.size_x
        size_y = metadata.images[0].pixels.size_y
        nSlices=metadata.images[0].pixels.size_z
        nChannels=metadata.images[0].pixels.size_c
        nFrames=metadata.images[0].pixels.size_t

        img=img.reshape(nFrames, nChannels, size_x, size_y)
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
                smoothness_flatfield=5,
                sort_intensity=False,
                fitting_mode="approximate",
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        # 2.2 Focus restoration in the reference channel
        img_logger.info(f"2.2 - Preprocessing image, Memory:{get_memory_usage()}")
        mean_intensity_1 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity before focus restoration:\n{np.round(mean_intensity_1, 3)}"
        )
        ip.img[:, 0, reference_channel]=self.bf_focus_restorer.predict(ip.img[:, 0, reference_channel])
        mean_intensity_2 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity after focus restoration:\n{np.round(mean_intensity_2, 3)}"
        )

        img_logger.info(f"2.2 - Preprocessing image, Memory:{get_memory_usage()}")
        mean_intensity_2 = np.mean(ip.img[:, 0, pi_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity before focus restoration:\n{np.round(mean_intensity_2, 3)}"
        )
        ip.img[:, 0, pi_channel]=self.fl_focus_restorer.predict(ip.img[:, 0, pi_channel])
        mean_intensity_2 = np.mean(ip.img[:, 0, pi_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity after focus restoration:\n{np.round(mean_intensity_2, 3)}"
        )
        # 2.3 Scale reference channel so that it works with previous classifer (relies on z-scaled images)
        mean_intensity_3 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity before channel intensity scaling:\n{np.round(mean_intensity_3, 3)}"
        )
        ip.scale_channel(range(nFrames), 0, nchannels=0)
        mean_intensity_2 = np.mean(ip.img[:, 0, reference_channel], axis=(1, 2))
        img_logger.info(
            f"Intensity after channel intensity scaling:\n{np.round(mean_intensity_2, 3)}"
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
        if is_cuda:
            img_logger.info(f"3.1 GPU  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        prob_map = self.image_segmentator.predict(
            ip.img[:, 0, reference_channel, :, :]
        )
        img_logger.info(f"3.1 - Segmentation completed! Memory:{get_memory_usage()}")
        if is_cuda:
            img_logger.info(f"3.1 GPU  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

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
        
        # Remove image-processor to release space
        del ip
        img_analyser.create_binary_mask()
        img_analyser.clean_binmask(min_pixel_size=20)
        img_analyser.get_labels()
        img_logger.info(f"{img_analyser.total_rois} objects found")

        # 3.3 Classify ROIs
        img_logger.info(f"3.2 - Classify ROIs, Memory:{get_memory_usage()}")
        if is_cuda:
            img_logger.info(f"3.2 GPU  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        #object_classes, labels=self.object_classifier.classify_rois(img_analyser.labeled_mask[:, 0,0], img_analyser.img[:, 0,0])
        object_classes, labels=self.batch_classify_rois(img_analyser, batch_size=5)
        if is_cuda:
            img_logger.info(f"3.2 GPU  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                
        # 4.1 Calc. measurements
        img_logger.info(f"4 - Starting measurements, Memory:{get_memory_usage()}")
        fl_prop = [
            "label",
            "centroid",
            "max_intensity",
            "min_intensity",
            "mean_intensity",
            "area",
        ]
        img_logger.info("Extracting fluorescent measurements")
        fl_measurements = img_analyser.get_roi_measurements(
            target_channel=1,
            properties=fl_prop,
            extra_properties=(roi_skewness, roi_std_dev),
        )
        fl_measurements["object_class"] = object_classes
        
        img_logger.info("Extracting time data")
        time_data = get_timestamps(metadata, timeformat="%Y-%m-%d %H:%M:%S")
        fl_measurements = pd.merge(fl_measurements, time_data, on="frame", how="left")
        fl_measurements = pd.merge(fl_measurements, bck_fl, on="frame", how="left")
        counts_per_frame = fl_measurements["frame"].value_counts().sort_index()
        img_logger.info(f"4 - Object counts per frame:\n{counts_per_frame.to_string()}")
        img_logger.info(f"4 - Measurements completed, Memory:{get_memory_usage()}")


        # 4.1 PI classification
        if self.pi_classifier is not None:
            img_logger.info(f"4.2 - Running PI classification, Memory:{get_memory_usage()}")
            predictions = self.pi_classifier.predict(
                fl_measurements[self.pi_classifier.feature_names_in_]
            )
            fl_measurements["pi_class"] = predictions
            fl_measurements['file'] = name
            try:
                d_summary = fl_measurements.groupby(['file', 'frame', 'channel', 'date_time', 'timestep', 'abslag_in_s', 'object_class']).agg(
                    total_count=('label', 'count'),
                    pi_class_neg=('pi_class', lambda x: (x == 'piNEG').sum()),
                    pi_class_pos=('pi_class', lambda x: (x == 'piPOS').sum()),
                    area_pineg=('area', lambda x: x[fl_measurements.loc[x.index, 'pi_class'] == 'piNEG'].sum()),
                    area_pipos=('area', lambda x: x[fl_measurements.loc[x.index, 'pi_class'] == 'piPOS'].sum()),
                    area_total=('area', 'sum'),
                ).reset_index()

                img_logger.info(f"Groupby operation completed successfully. Shape of d_summary: {d_summary.shape}")
            except Exception as e:
                img_logger.error(f"Error during groupby operation: {str(e)}")
                img_logger.error(f"Columns in fl_measurements: {fl_measurements.columns}")
                img_logger.error(f"Unique values in 'pi_class': {fl_measurements['pi_class'].unique()}")
                d_summary=pd.DataFrame()
            img_logger.info(f"d_summary created successfully. Memory usage: {get_memory_usage()}")
        else:
            d_summary = pd.DataFrame()

        # 5. Export data
        export_path = os.path.join(self.output_path, name)
        img_logger.info(f"5 - Writing data to {export_path}")

        fl_measurements.to_csv(export_path + "_fl.csv")
        d_summary.to_csv(export_path + "_summary.csv")

        if export_labeled_mask:
            class_to_id = {
                'single-cell': 0,
                'clump': 1,
                'noise': 2,
                'off-focus': 3,
                'joint-cell': 4
            }

            # Create a mapping from original label IDs to new class IDs
            label_to_class_id = {label: class_to_id[class_name] + 1 for label, class_name in zip(labels, object_classes)}
            vectorized_map = np.vectorize(lambda x: label_to_class_id.get(x, 0))
            new_labeled_mask = vectorized_map(img_analyser.labeled_mask[:, 0, 0])
            labs_8bit = new_labeled_mask.astype(np.uint8)
            tifffile.imwrite(export_path + "_labels.tiff", labs_8bit)
        if export_aligned_image:
            image_8bit = convert_image(img_analyser.img, np.uint8)
            tifffile.imwrite(export_path + "_transformed.tiff", image_8bit, imagej=True)

        img_logger.info(f"Analysis completed for {movie_name}, Memory:{get_memory_usage()}")
        del prob_map, img, fl_measurements, d_summary, img_analyser
        gc.collect()
        empty_gpu_cache(get_device())
        img_logger.info(f"Garbage collection completed, Memory:{get_memory_usage()}")
        
        self.remove_logger(img_logger)
        
        return name
    
    def batch_classify_rois(self, img_analyser, batch_size=5):
        labeled_mask = img_analyser.labeled_mask[:, 0, 0]#.get()
        img = img_analyser.img[:, 0, 0]#.get()
        
        n_frames = labeled_mask.shape[0]
        all_object_classes = []
        all_labels = []
        
        for start_frame in range(0, n_frames, batch_size):
            end_frame = min(start_frame + batch_size, n_frames)
            
            # Extract batch of frames
            batch_labeled_mask = labeled_mask[start_frame:end_frame]
            batch_img = img[start_frame:end_frame]
            
            # Classify the batch
            batch_classes, batch_labels = self.object_classifier.classify_rois(batch_labeled_mask, batch_img)
            
            all_object_classes.extend(batch_classes)
            all_labels.extend(batch_labels)
        
        return all_object_classes, all_labels