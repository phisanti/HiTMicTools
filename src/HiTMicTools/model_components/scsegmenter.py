import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from rfdetr import RFDETRSegPreview

from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.resource_management.sysutils import get_device


@dataclass
class SegmentationBatch:
    """Container used internally to keep track of tile metadata during inference."""

    tiles: List[torch.Tensor]
    offsets: List[Tuple[int, int]]
    valid_shapes: List[Tuple[int, int]]
    image_shape: Tuple[int, int]


class ScSegmenter(BaseModel):
    """
    Single-cell instance segmenter based on RF-DETR-Segm with custom sliding-window inference.

    This class handles high-resolution microscopy frames by tiling them into
    RF-DETR-sized crops, forwarding each crop through the detector, and merging
    the results with batched NMS to recover full-frame instance segmentations.

    Unlike the OofDetector, this model returns both bounding boxes and instance masks,
    performing simultaneous detection, segmentation, and classification of single cells.

    Attributes:
        model: RF-DETR segmentation model instance
        device: Torch device (CPU/CUDA)
        tile_size: Square tile edge length for sliding window
        overlap_ratio: Fractional overlap between adjacent tiles
        score_threshold: Minimum confidence for detections
        nms_iou: IoU threshold for non-maximum suppression
        class_dict: Mapping from class indices to class names
    """

    def __init__(
        self,
        model_path: str,
        patch_size: int = 560,
        overlap_ratio: float = 0.25,
        score_threshold: float = 0.5,
        nms_iou: float = 0.5,
        class_dict: Optional[dict] = None,
        model_type: str = "rfdetrsegpreview",
    ) -> None:
        """
        Initialize the single-cell segmenter.

        Args:
            model_path: Filesystem path to a RF-DETR checkpoint (.pth).
            patch_size: Square tile edge length passed to RF-DETR.
            overlap_ratio: Fractional overlap between adjacent tiles.
            score_threshold: Minimum confidence kept from per-tile detections.
            nms_iou: IoU threshold for per-class non-maximum suppression.
            class_dict: Dictionary mapping class indices to names (e.g., {0: 'single-cell', 1: 'clump'}).
                If provided, num_classes is derived from its length. If None, inferred from checkpoint.
            model_type: Identifier for the detector backbone to instantiate.
        """
        assert 0 <= overlap_ratio < 1, "overlap_ratio must be in [0, 1)."
        assert patch_size > 0, "patch_size must be positive."

        self.device = get_device()
        if self.device.type == "mps":
            warnings.warn(
                "ScSegmenter falling back to CPU because RF-DETR backbone "
                "uses ops unsupported on MPS.",
                RuntimeWarning,
            )
            self.device = torch.device("cpu")

        self.tile_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.class_dict = class_dict

        if model_type.lower() != "rfdetrsegpreview":
            raise ValueError(
                f"Unsupported segmenter type '{model_type}'. "
                "Currently only 'rfdetrsegpreview' is supported."
            )

        # Determine num_classes from class_dict or infer from checkpoint
        num_classes = None
        if class_dict:
            num_classes = len(class_dict)
        else:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            class_bias = checkpoint["model"]["class_embed.bias"]
            num_classes = class_bias.shape[0] - 1

        self.model = RFDETRSegPreview(
            pretrain_weights=model_path,
            num_classes=num_classes,
            device=self.device.type,
        )
        # RFDETR keeps its own device bookkeeping; make sure the model graph
        # is in eval mode to prevent dropout/batch-norm updates.
        self.model.model.model.eval()

        # Compile the inner model for optimized inference
        self.model.model.model = torch.compile(
            self.model.model.model, mode="max-autotune"
        )

    def predict(
        self,
        image: Union[np.ndarray, torch.Tensor],
        channel_index: int = 0,
        temporal_buffer_size: int = 8,
        spatial_batch_size: int = 32,
        normalize_to_255: bool = True,
        score_threshold: Optional[float] = None,
        output_shape: str = "HW",
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run 4D sliding-window inference with temporal buffering on microscopy images.

        This method efficiently processes time-series microscopy data by:
        1. Loading temporal chunks into GPU memory (controlled by temporal_buffer_size)
        2. Interleaving spatial tiles from multiple frames for maximum GPU utilization
        3. Processing tiles in batches with cross-frame batching
        4. Releasing processed frames from GPU to manage memory

        The method automatically handles both single-frame and multi-frame inputs,
        and performs channel padding (grayscale → RGB) internally.

        Args:
            image: Input array with shape:
                - [T, H, W]: Multi-frame grayscale (most common from pipelines)
                - [T, C, H, W]: Multi-frame multi-channel
                - [H, W]: Single frame grayscale
                - [C, H, W]: Single frame multi-channel
            channel_index: Channel to segment (default: 0 for grayscale)
            temporal_buffer_size: Number of frames to keep in GPU memory at once.
                Larger values increase GPU memory usage but reduce CPU↔GPU transfers.
                Recommended: 4-8 for typical workloads, 2-4 for memory-constrained GPUs.
            spatial_batch_size: Number of spatial tiles to process in parallel.
                Larger values improve GPU utilization but increase memory usage.
                Recommended: 16-32 for typical GPUs.
            normalize_to_255: Whether to min-max normalize the selected channel
                to [0, 1] range. Set to False if image is already preprocessed.
            score_threshold: Optional override for detection confidence threshold.
                If None, uses the threshold set during initialization.
            output_shape: Output dimension ordering for masks. Options:
                - "HW": Height × Width (default, standard image convention)
                - "WH": Width × Height (HiTMicTools TSCXY convention compatibility)
                For multi-frame output, this applies to the last two dimensions of [T, *, *]

        Returns:
            Tuple of:
                - labeled_masks: [T, H, W] or [T, W, H] array of stacked instance masks
                  (or [H, W]/[W, H] for single frame), depending on output_shape parameter
                - bboxes_list: List of [N_t, 4] bbox arrays per frame (xyxy format)
                - class_ids_list: List of [N_t] class ID arrays per frame
                - scores_list: List of [N_t] confidence score arrays per frame

        Examples:
            >>> # Multi-frame input with HW output (standard)
            >>> frames = ip.img[:, 0, 0, :, :]  # Shape: [T, H, W]
            >>> masks, bboxes, classes, scores = segmenter.predict(
            ...     frames,
            ...     channel_index=0,
            ...     temporal_buffer_size=8,
            ...     spatial_batch_size=32,
            ...     normalize_to_255=False,
            ...     output_shape="HW"  # Returns [T, H, W]
            ... )

            >>> # Multi-frame input with WH output (HiTMicTools TSCXY compatibility)
            >>> frames = ip.img[:, 0, 0, :, :]  # Shape: [T, X, Y] in TSCXY convention
            >>> masks, bboxes, classes, scores = segmenter.predict(
            ...     frames,
            ...     output_shape="WH"  # Returns [T, X, Y] matching ip.img convention
            ... )

            >>> # Single frame input (backward compatible)
            >>> frame = ip.img[0, 0, 0, :, :]  # Shape: [H, W]
            >>> mask, bboxes, classes, scores = segmenter.predict(
            ...     frame,
            ...     temporal_buffer_size=1,
            ...     spatial_batch_size=32
            ... )
        """
        if spatial_batch_size <= 0:
            raise ValueError("spatial_batch_size must be positive.")
        if temporal_buffer_size <= 0:
            raise ValueError("temporal_buffer_size must be positive.")
        if output_shape not in ["HW", "WH"]:
            raise ValueError(f"output_shape must be 'HW' or 'WH', got '{output_shape}'")

        # Use provided threshold or fall back to instance threshold
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # 1. Prepare input - convert to [T, H, W] format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image, is_single_frame = self._reshape_input(image, channel_index)

        # 2. Process in temporal buffers
        num_frames = image.shape[0]
        all_labeled_masks = []
        all_bboxes = []
        all_class_ids = []
        all_scores = []

        for buffer_start in range(0, num_frames, temporal_buffer_size):
            buffer_end = min(buffer_start + temporal_buffer_size, num_frames)

            # Load temporal buffer to GPU
            buffer_frames = image[buffer_start:buffer_end].to(self.device)

            # Process frames in the buffer with cross-frame tile batching
            buffer_results = self._process_temporal_buffer(
                buffer_frames,
                spatial_batch_size=spatial_batch_size,
                normalize_to_255=normalize_to_255,
                score_threshold=threshold,
            )

            # Aggregate results
            for labeled_mask, bboxes, class_ids, scores in buffer_results:
                all_labeled_masks.append(labeled_mask)
                all_bboxes.append(bboxes)
                all_class_ids.append(class_ids)
                all_scores.append(scores)

            # Explicitly free buffer from GPU
            del buffer_frames
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # 3. Format and return output
        return self._format_output(
            all_labeled_masks, all_bboxes, all_class_ids, all_scores,
            is_single_frame, output_shape
        )

    def _reshape_input(
        self,
        image: torch.Tensor,
        channel_index: int,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Reshape input image to [T, H, W] format and detect if single frame.

        Args:
            image: Input tensor with variable dimensions
            channel_index: Channel to extract if multi-channel

        Returns:
            Tuple of:
                - image: [T, H, W] tensor ready for processing
                - is_single_frame: Boolean indicating if input was a single frame

        Raises:
            ValueError: If input dimensions are not 2D, 3D, or 4D
            IndexError: If channel_index is out of bounds
        """
        image = image.squeeze()
        is_single_frame = False

        if image.ndim == 2:
            # [H, W] → [1, 1, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
            is_single_frame = True
        elif image.ndim == 3:
            # Could be [C, H, W] or [T, H, W]
            # Assume [T, H, W] (most common from pipelines)
            # If user has [C, H, W], they should use channel_index appropriately
            image = image.unsqueeze(1)  # [T, H, W] → [T, 1, H, W]
            if image.shape[0] == 1:
                is_single_frame = True
        elif image.ndim == 4:
            # [T, C, H, W] - already in correct format
            if image.shape[0] == 1:
                is_single_frame = True
        else:
            raise ValueError(
                f"Unsupported image dimensions: expected 2D, 3D, or 4D input, got shape {image.shape}."
            )

        # Validate channel index
        num_channels = image.shape[1]
        if channel_index >= num_channels:
            raise IndexError(
                f"channel_index {channel_index} out of bounds for image with {num_channels} channels."
            )

        # Extract target channel: [T, C, H, W] → [T, H, W]
        image = image[:, channel_index, :, :].to(dtype=torch.float32)

        return image, is_single_frame

    def _format_output(
        self,
        all_labeled_masks: List[np.ndarray],
        all_bboxes: List[np.ndarray],
        all_class_ids: List[np.ndarray],
        all_scores: List[np.ndarray],
        is_single_frame: bool,
        output_shape: str,
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]],
               Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
        """
        Format output masks and detections based on single/multi-frame and output_shape.

        Args:
            all_labeled_masks: List of [H, W] masks from each frame
            all_bboxes: List of bbox arrays from each frame
            all_class_ids: List of class ID arrays from each frame
            all_scores: List of score arrays from each frame
            is_single_frame: Whether input was a single frame
            output_shape: "HW" or "WH" dimension ordering

        Returns:
            Tuple of:
                - labeled_masks: Single mask [H/W, W/H] or stacked [T, H/W, W/H]
                - bboxes: Single array or list of arrays
                - class_ids: Single array or list of arrays
                - scores: Single array or list of arrays
        """
        if is_single_frame:
            # Return single frame format for backward compatibility
            mask = all_labeled_masks[0]
            if output_shape == "WH":
                mask = mask.T  # [H, W] → [W, H]
            return mask, all_bboxes[0], all_class_ids[0], all_scores[0]
        else:
            # Stack masks for multi-frame output
            stacked_masks = np.stack(all_labeled_masks, axis=0)  # [T, H, W]
            if output_shape == "WH":
                # Transpose last two dimensions: [T, H, W] → [T, W, H]
                stacked_masks = np.transpose(stacked_masks, (0, 2, 1))
            return stacked_masks, all_bboxes, all_class_ids, all_scores

    def _process_temporal_buffer(
        self,
        buffer_frames: torch.Tensor,
        spatial_batch_size: int,
        normalize_to_255: bool,
        score_threshold: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Process a temporal buffer of frames with cross-frame tile batching.

        This method implements the advanced batching strategy:
        1. Pre-tiles all frames in the buffer
        2. Interleaves tiles from different frames for efficient GPU batching
        3. Processes tiles from multiple frames in the same batch
        4. Demultiplexes detections back to their respective frames

        This maximizes GPU utilization by allowing RF-DETR to process
        tile_0_frame_0, tile_0_frame_1, ... tile_0_frame_N in the same batch.

        Args:
            buffer_frames: [B, H, W] tensor of frames to process
            spatial_batch_size: Number of tiles to process in parallel
            normalize_to_255: Whether to normalize frames
            score_threshold: Detection confidence threshold

        Returns:
            List of (labeled_mask, bboxes, class_ids, scores) tuples, one per frame
        """
        buffer_size = buffer_frames.shape[0]

        # 1. Prepare all frames and create tiles
        all_batches = []
        for frame_idx in range(buffer_size):
            frame_tensor = self._prepare_frame_tensor(
                buffer_frames[frame_idx],
                normalize_to_255=normalize_to_255,
            )
            batch = self._create_tiles(frame_tensor)
            all_batches.append(batch)

        # 2. Interleave tiles from all frames for better batching
        # Strategy: Process corresponding tiles from all frames together
        # e.g., [tile_0_frame_0, tile_0_frame_1, tile_0_frame_2, ...]
        # This allows the GPU to batch similar spatial regions across time
        mega_tiles = []
        tile_to_frame_map = []

        max_tiles = max(len(batch.tiles) for batch in all_batches)

        for tile_idx in range(max_tiles):
            for frame_idx, batch in enumerate(all_batches):
                if tile_idx < len(batch.tiles):
                    mega_tiles.append(batch.tiles[tile_idx])
                    tile_to_frame_map.append((frame_idx, tile_idx))

        # 3. Process mega-batch with spatial batching
        all_detections = []
        for start in range(0, len(mega_tiles), spatial_batch_size):
            batch_tiles = [tile.cpu() for tile in mega_tiles[start:start + spatial_batch_size]]
            predictions = self.model.predict(batch_tiles, threshold=score_threshold)
            if not isinstance(predictions, list):
                predictions = [predictions]
            all_detections.extend(predictions)

        # 4. Demultiplex detections back to frames
        # Create a list of detection lists, one per frame
        frame_detections = [[] for _ in range(buffer_size)]
        for detection, (frame_idx, tile_idx) in zip(all_detections, tile_to_frame_map):
            frame_detections[frame_idx].append(detection)

        # 5. Merge detections for each frame
        results = []
        for frame_idx, detections in enumerate(frame_detections):
            batch = all_batches[frame_idx]
            labeled_mask, boxes, class_ids, scores = self._merge_detections(batch, detections)
            results.append((labeled_mask, boxes, class_ids, scores))

        return results

    def _prepare_frame_tensor(
        self,
        frame: torch.Tensor,
        normalize_to_255: bool,
    ) -> torch.Tensor:
        """
        Convert a single-channel frame to RGB tensor ready for RF-DETR.

        Args:
            frame: [H, W] tensor, single channel, float32
            normalize_to_255: Whether to apply min-max normalization

        Returns:
            [3, H, W] tensor ready for RF-DETR inference
        """
        # Ensure frame is [1, H, W] for processing
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)

        # Apply min-max normalization if requested
        # This is typically needed for raw microscopy images but NOT for preprocessed images
        if normalize_to_255:
            frame = frame - frame.amin(dim=(-2, -1), keepdim=True)
            max_val = frame.amax(dim=(-2, -1), keepdim=True)
            frame = torch.where(
                max_val > 0, frame / max_val, torch.zeros_like(frame)
            )

        # Pad grayscale to RGB (replicate channel 3 times)
        frame = frame.repeat(3, 1, 1)  # [1, H, W] → [3, H, W]

        # Verify values are in [0, 1] range as required by RF-DETR
        if frame.max() > 1.0:
            frame = frame.clamp(0, 1)

        return frame

    def _create_tiles(self, image_tensor: torch.Tensor) -> SegmentationBatch:
        """Decompose full image into overlapping tiles for sliding-window inference."""
        _, height, width = image_tensor.shape
        padded_tensor = self._pad_if_needed(image_tensor)
        _, padded_h, padded_w = padded_tensor.shape

        # DEBUG: Log padding information
        print(f"[ScSegmenter] Original image: {height}x{width}, Padded: {padded_h}x{padded_w}")

        step = max(int(self.tile_size * (1 - self.overlap_ratio)), 1)
        x_positions = self._compute_positions(padded_w, step)
        y_positions = self._compute_positions(padded_h, step)

        tiles: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        valid_shapes: List[Tuple[int, int]] = []

        for y in y_positions:
            for x in x_positions:
                crop = padded_tensor[:, y : y + self.tile_size, x : x + self.tile_size]
                # DEBUG: Check tile shape
                if crop.shape[1] != self.tile_size or crop.shape[2] != self.tile_size:
                    print(f"[ScSegmenter] WARNING: Tile at ({x}, {y}) has shape {crop.shape}, expected [{crop.shape[0]}, {self.tile_size}, {self.tile_size}]")
                tiles.append(crop)

                valid_h = min(self.tile_size, height - y) if y < height else 0
                valid_w = min(self.tile_size, width - x) if x < width else 0
                valid_shapes.append((max(valid_h, 0), max(valid_w, 0)))
                offsets.append((x, y))

        return SegmentationBatch(
            tiles=tiles,
            offsets=offsets,
            valid_shapes=valid_shapes,
            image_shape=(height, width),
        )

    def _pad_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad image with mean pixel value to ensure all tiles are exactly tile_size × tile_size.

        This calculates padding needed so that the last tile position + tile_size
        doesn't exceed the padded dimensions, preventing partial tiles.
        """
        _, height, width = tensor.shape
        step = max(int(self.tile_size * (1 - self.overlap_ratio)), 1)

        # Calculate the required padded dimensions
        # We need to ensure that the last tile starting position + tile_size fits exactly
        def calc_padded_size(length: int) -> int:
            if length <= self.tile_size:
                return self.tile_size

            # Calculate number of steps needed
            num_steps = (length - self.tile_size + step - 1) // step
            # Last position is at num_steps * step
            last_position = num_steps * step
            # Required size is last_position + tile_size
            required_size = last_position + self.tile_size
            return required_size

        required_height = calc_padded_size(height)
        required_width = calc_padded_size(width)

        pad_bottom = required_height - height
        pad_right = required_width - width

        if pad_bottom == 0 and pad_right == 0:
            return tensor

        # Calculate mean pixel value for padding
        mean_value = tensor.mean()

        # Pad with mean value instead of zeros
        return F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=mean_value.item())

    def _compute_positions(self, length: int, step: int) -> List[int]:
        """Calculate tile starting positions along one dimension."""
        if length <= self.tile_size:
            return [0]

        positions = list(range(0, length - self.tile_size + 1, step))
        if positions[-1] != length - self.tile_size:
            positions.append(length - self.tile_size)
        return positions

    def _infer_tiles(
        self,
        batch: SegmentationBatch,
        batch_size: int,
        threshold: float,
    ) -> Sequence:
        """Run RF-DETR inference on all tiles in batches."""
        detections = []

        for start in range(0, len(batch.tiles), batch_size):
            batch_tiles = [
                tile.cpu() for tile in batch.tiles[start : start + batch_size]
            ]
            predictions = self.model.predict(
                batch_tiles, threshold=threshold
            )
            if not isinstance(predictions, list):
                predictions = [predictions]

            detections.extend(predictions)

        return detections

    def _merge_detections(
        self,
        batch: SegmentationBatch,
        detections: Sequence,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge per-tile detections into a full-frame labeled mask and detection arrays.

        This method:
        1. Collects all bounding boxes, scores, and classes from tiles
        2. Adjusts coordinates to full-frame space
        3. Applies batched NMS to remove duplicate detections
        4. Stitches instance masks into a single labeled mask
        """
        boxes: List[torch.Tensor] = []
        class_ids: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        offsets_list: List[Tuple[int, int]] = []

        for det, (offset_x, offset_y), (valid_h, valid_w) in zip(
            detections, batch.offsets, batch.valid_shapes
        ):
            if det is None or len(det) == 0:
                continue

            tile_boxes = torch.from_numpy(det.xyxy)
            tile_scores = torch.from_numpy(det.confidence)
            tile_classes = torch.from_numpy(det.class_id)

            # Adjust bounding box coordinates to full-frame space
            tile_boxes[:, 0::2] += offset_x
            tile_boxes[:, 1::2] += offset_y

            # Clamp to valid region
            if valid_h > 0 and valid_w > 0:
                max_x = offset_x + valid_w
                max_y = offset_y + valid_h
                tile_boxes[:, 0::2] = tile_boxes[:, 0::2].clamp(max=max_x)
                tile_boxes[:, 1::2] = tile_boxes[:, 1::2].clamp(max=max_y)

            boxes.append(tile_boxes)
            scores.append(tile_scores)
            class_ids.append(tile_classes)

            # Store masks (if available) with their tile offsets
            # Note: supervision uses 'mask' (singular) not 'masks' (plural)
            if hasattr(det, 'mask') and det.mask is not None and len(det.mask) > 0:
                tile_masks = torch.from_numpy(det.mask)
                masks.append(tile_masks)
                offsets_list.extend([(offset_x, offset_y)] * len(tile_masks))

        if not boxes:
            height, width = batch.image_shape
            return (
                np.zeros((height, width), dtype=np.int32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        boxes_tensor = torch.cat(boxes, dim=0).float()
        scores_tensor = torch.cat(scores, dim=0).float()
        classes_tensor = torch.cat(class_ids, dim=0).long()

        # Clamp to image bounds
        height, width = batch.image_shape
        boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, width)
        boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, height)

        # Apply per-class NMS to remove duplicates from overlapping tiles
        #
        # Why per-class NMS instead of batched_nms?
        # - batched_nms expects a fixed number of detections per image (designed for batch processing)
        # - Here we're processing tiles from a SINGLE image, where the number of detections varies
        # - Using regular nms per class allows flexible detection counts while preventing
        #   cross-class suppression (e.g., a high-confidence "single-cell" shouldn't suppress
        #   a lower-confidence "clump" that happens to overlap spatially)
        keep_indices_list = []
        for class_id in torch.unique(classes_tensor):
            class_mask = classes_tensor == class_id
            class_boxes = boxes_tensor[class_mask]
            class_scores = scores_tensor[class_mask]

            # Get indices within this class that survive NMS
            class_keep = nms(class_boxes, class_scores, self.nms_iou)

            # Map back to original indices
            original_indices = torch.where(class_mask)[0]
            keep_indices_list.append(original_indices[class_keep])

        # Combine all kept indices and sort them to maintain consistent ordering
        if keep_indices_list:
            keep_indices = torch.cat(keep_indices_list).sort()[0]
        else:
            keep_indices = torch.tensor([], dtype=torch.long)

        boxes_np = boxes_tensor[keep_indices].numpy()
        classes_np = classes_tensor[keep_indices].numpy()
        scores_np = scores_tensor[keep_indices].numpy()

        # Stitch masks into labeled mask
        if masks:
            masks_tensor = torch.cat(masks, dim=0)
            labeled_mask, mask_to_detection_map = self._stitch_masks(
                masks_tensor, keep_indices, offsets_list, batch.image_shape
            )

            # Filter class_ids and scores to match only the instances that made it into the labeled mask
            # mask_to_detection_map gives us the index in the NMS-filtered detections for each mask instance
            if len(mask_to_detection_map) > 0:
                classes_np = classes_np[mask_to_detection_map]
                scores_np = scores_np[mask_to_detection_map]
                boxes_np = boxes_np[mask_to_detection_map]
        else:
            # If no masks available, create empty labeled mask
            labeled_mask = np.zeros(batch.image_shape, dtype=np.int32)

        # Note: labeled_mask is in [H, W] format (standard image convention)
        # The predict() method will transpose to [W, H] if output_shape="WH" is requested
        # for HiTMicTools TSCXY convention compatibility

        # DEBUG: Log detection counts
        num_detections = len(boxes_np)
        unique_instances = len(np.unique(labeled_mask)) - 1  # Subtract 1 to exclude background (0)
        print(f"[ScSegmenter] Detections: {num_detections} bboxes, {unique_instances} unique mask instances")
        if num_detections > 0:
            print(f"[ScSegmenter] Score range: [{scores_np.min():.3f}, {scores_np.max():.3f}]")
            print(f"[ScSegmenter] Class distribution: {np.bincount(classes_np.astype(int))}")

        return labeled_mask, boxes_np, classes_np, scores_np

    def _stitch_masks(
        self,
        masks: torch.Tensor,
        keep_indices: torch.Tensor,
        offsets: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stitch per-tile instance masks into a single full-frame labeled mask.

        Args:
            masks: [N, H_tile, W_tile] tensor of binary masks from all tiles
            keep_indices: Indices of detections that survived NMS
            offsets: List of (x, y) offsets for each mask
            image_shape: (H, W) of the full frame

        Returns:
            labeled_mask: [H, W] array with unique integer labels per instance
            mask_to_detection_map: [M] array mapping each unique mask instance to its detection index
        """
        height, width = image_shape
        labeled_mask = np.zeros((height, width), dtype=np.int32)

        # Only keep masks that survived NMS
        kept_masks = masks[keep_indices]
        kept_offsets = [offsets[i] for i in keep_indices.cpu().numpy()]

        # Track which detection index corresponds to each mask instance that gets written
        mask_to_detection_map = []

        for detection_idx, (mask, (offset_x, offset_y)) in enumerate(
            zip(kept_masks, kept_offsets)
        ):
            mask_np = mask.cpu().numpy()
            mask_h, mask_w = mask_np.shape

            # Calculate the valid region within the full frame
            y_start = offset_y
            y_end = min(offset_y + mask_h, height)
            x_start = offset_x
            x_end = min(offset_x + mask_w, width)

            # Crop mask to valid region
            mask_crop_h = y_end - y_start
            mask_crop_w = x_end - x_start
            mask_crop = mask_np[:mask_crop_h, :mask_crop_w]

            # Convert to binary
            binary_mask = mask_crop > 0.5

            # Check if this mask actually adds any pixels (not completely overlapped)
            roi = labeled_mask[y_start:y_end, x_start:x_end]
            new_pixels = binary_mask & (roi == 0)

            if new_pixels.any():
                # Assign a new label ID for this instance
                label_id = len(mask_to_detection_map) + 1
                labeled_mask[y_start:y_end, x_start:x_end] = np.where(
                    new_pixels, label_id, roi
                )
                # Track which detection this label corresponds to
                mask_to_detection_map.append(detection_idx)

        return labeled_mask, np.array(mask_to_detection_map, dtype=np.int32)
