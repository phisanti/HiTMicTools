import logging
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from rfdetr import RFDETRBase

from HiTMicTools.model_components.base_model import BaseModel
from HiTMicTools.resource_management.sysutils import get_device


@dataclass
class DetectionBatch:
    """Container used internally to keep track of tile metadata."""

    tiles: List[torch.Tensor]
    offsets: List[Tuple[int, int]]
    valid_shapes: List[Tuple[int, int]]
    image_shape: Tuple[int, int]


class OofDetector(BaseModel):
    """
    Out-of-focus detector based on RFDETR with custom sliding-window inference.

    This class handles high-resolution microscopy frames by tiling them into
    RF-DETR-sized crops, forwarding each crop through the detector, and merging
    the results with batched NMS to recover full-frame detections.
    """

    def __init__(
        self,
        model_path: str,
        patch_size: int = 560,
        overlap_ratio: float = 0.25,
        score_threshold: float = 0.5,
        nms_iou: float = 0.5,
        class_dict: Optional[dict] = None,
        model_type: str = "rfdetrbase",
    ) -> None:
        """
        Args:
            model_path: Filesystem path to a RF-DETR checkpoint (.pth).
            patch_size: Square tile edge length passed to RF-DETR.
            overlap_ratio: Fractional overlap between adjacent tiles.
            score_threshold: Minimum confidence kept from per-tile detections.
            nms_iou: IoU threshold for per-class non-maximum suppression.
            class_dict: Dictionary mapping class names to indices. If provided,
                num_classes is derived from its length. If None, inferred from checkpoint.
            model_type: Identifier for the detector backbone to instantiate.
        """
        assert 0 <= overlap_ratio < 1, "overlap_ratio must be in [0, 1)."
        assert patch_size > 0, "patch_size must be positive."

        self.device = get_device()
        if self.device.type == "mps":
            warnings.warn(
                "OofDetector falling back to CPU because RF-DETR backbone "
                "uses ops unsupported on MPS.",
                RuntimeWarning,
            )
            self.device = torch.device("cpu")

        self.tile_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou

        if model_type.lower() != "rfdetrbase":
            raise ValueError(
                f"Unsupported detector type '{model_type}'. "
                "Currently only 'rfdetrbase' is supported."
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

        self.model = RFDETRBase(
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
        frame_index: int = 0,
        channel_index: int = 0,
        pad_to_rgb: bool = True,
        normalize_to_255: bool = True,
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run sliding-window inference on a microscopy frame.

        Args:
            image: Input array with shape (C, H, W) or (T, C, H, W).
            frame_index: Frame to select when image has temporal dimension.
            channel_index: Channel to feed into the detector.
            pad_to_rgb: If True, replicates the channel to produce 3-channel RGB.
            normalize_to_255: Whether to min-max normalise the selected channel
                to [0, 255] before converting to float32 in [0, 1].
            batch_size: Number of tiles inferred together.

        Returns:
            Tuple of (bboxes[N,4], class_ids[N], confidences[N]).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        image_tensor = self._prepare_image_tensor(
            image=image,
            frame_index=frame_index,
            channel_index=channel_index,
            pad_to_rgb=pad_to_rgb,
            normalize_to_255=normalize_to_255,
        )

        batch = self._create_tiles(image_tensor)

        detections = self._infer_tiles(batch, batch_size=batch_size)

        boxes, class_ids, scores = self._merge_detections(batch, detections)

        return boxes, class_ids, scores

    def _prepare_image_tensor(
        self,
        image: Union[np.ndarray, torch.Tensor],
        frame_index: int,
        channel_index: int,
        pad_to_rgb: bool,
        normalize_to_255: bool,
    ) -> torch.Tensor:
        """
        Convert the user supplied array into a normalized 3×HxW tensor ready for tiling.

        Args:
            image: Input array/tensor that may contain time and channel dimensions.
            frame_index: Frame index used when a temporal axis is present.
            channel_index: Channel sent to the detector.
            pad_to_rgb: If True replicate a single channel into 3 planes.
            normalize_to_255: When True the channel is min-max normalized.
        """
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        else:
            tensor = image

        tensor = tensor.squeeze()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor[frame_index]
        elif tensor.ndim != 3:
            raise ValueError(
                f"Unsupported image dimensions: expected 2D, 3D, or 4D input, got shape {tensor.shape}."
            )

        if channel_index >= tensor.shape[0]:
            raise IndexError(
                f"channel_index {channel_index} out of bounds for image with {tensor.shape[0]} channels."
            )

        tensor = tensor[channel_index].unsqueeze(0).to(dtype=torch.float32)

        if normalize_to_255:
            tensor = tensor - tensor.amin(dim=(-2, -1), keepdim=True)
            max_val = tensor.amax(dim=(-2, -1), keepdim=True)
            tensor = torch.where(
                max_val > 0, tensor / max_val, torch.zeros_like(tensor)
            )

        if pad_to_rgb:
            tensor = tensor.repeat(3, 1, 1)

        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        return tensor

    def _create_tiles(self, image_tensor: torch.Tensor) -> DetectionBatch:
        """Tile the preprocessed tensor and track offsets plus valid slice sizes."""
        _, height, width = image_tensor.shape
        padded_tensor = self._pad_if_needed(image_tensor)
        _, padded_h, padded_w = padded_tensor.shape

        step = max(int(self.tile_size * (1 - self.overlap_ratio)), 1)
        x_positions = self._compute_positions(padded_w, step)
        y_positions = self._compute_positions(padded_h, step)

        tiles: List[torch.Tensor] = []
        offsets: List[Tuple[int, int]] = []
        valid_shapes: List[Tuple[int, int]] = []

        for y in y_positions:
            for x in x_positions:
                crop = padded_tensor[:, y : y + self.tile_size, x : x + self.tile_size]
                tiles.append(crop)

                valid_h = min(self.tile_size, height - y) if y < height else 0
                valid_w = min(self.tile_size, width - x) if x < width else 0
                valid_shapes.append((max(valid_h, 0), max(valid_w, 0)))
                offsets.append((x, y))

        return DetectionBatch(
            tiles=tiles,
            offsets=offsets,
            valid_shapes=valid_shapes,
            image_shape=(height, width),
        )

    def _pad_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply zero-padding when the image is smaller than the tile size."""
        _, height, width = tensor.shape
        pad_bottom = max(self.tile_size - height, 0)
        pad_right = max(self.tile_size - width, 0)

        if pad_bottom == 0 and pad_right == 0:
            return tensor

        return F.pad(tensor, (0, pad_right, 0, pad_bottom))

    def _compute_positions(self, length: int, step: int) -> List[int]:
        """Return all tile start indices for a given axis length and sliding step."""
        if length <= self.tile_size:
            return [0]

        positions = list(range(0, length - self.tile_size + 1, step))
        if positions[-1] != length - self.tile_size:
            positions.append(length - self.tile_size)
        return positions

    def _infer_tiles(
        self,
        batch: DetectionBatch,
        batch_size: int,
    ) -> Sequence:
        """
        Run batch inference over the prepared tiles and return the raw RF-DETR outputs.

        Args:
            batch: Container returned by `_create_tiles`.
            batch_size: Number of tiles to process together.
        """
        detections = []

        for start in range(0, len(batch.tiles), batch_size):
            batch_tiles = [
                tile.cpu().clamp(0, 1) for tile in batch.tiles[start : start + batch_size]
            ]
            predictions = self.model.predict(
                batch_tiles, threshold=self.score_threshold
            )
            if not isinstance(predictions, list):
                predictions = [predictions]
            detections.extend(predictions)

        return detections

    def _merge_detections(
        self,
        batch: DetectionBatch,
        detections: Sequence,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge per-tile detections back into full-frame coordinates and run NMS.

        Args:
            batch: Batch metadata describing offsets and valid regions.
            detections: Raw predictions generated by `_infer_tiles`.

        Returns:
            Tuple of numpy arrays with boxes, class ids, and confidence scores.
        """
        boxes: List[torch.Tensor] = []
        class_ids: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []

        for det, (offset_x, offset_y), (valid_h, valid_w) in zip(
            detections, batch.offsets, batch.valid_shapes
        ):
            if det is None or len(det) == 0:
                continue

            tile_boxes = torch.from_numpy(det.xyxy)
            tile_scores = torch.from_numpy(det.confidence)
            tile_classes = torch.from_numpy(det.class_id)

            tile_boxes[:, 0::2] += offset_x
            tile_boxes[:, 1::2] += offset_y

            if valid_h > 0 and valid_w > 0:
                max_x = offset_x + valid_w
                max_y = offset_y + valid_h
                tile_boxes[:, 0::2] = tile_boxes[:, 0::2].clamp(max=max_x)
                tile_boxes[:, 1::2] = tile_boxes[:, 1::2].clamp(max=max_y)

            boxes.append(tile_boxes)
            scores.append(tile_scores)
            class_ids.append(tile_classes)

        if not boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        boxes_tensor = torch.cat(boxes, dim=0).float()
        scores_tensor = torch.cat(scores, dim=0).float()
        classes_tensor = torch.cat(class_ids, dim=0).long()

        height, width = batch.image_shape
        boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, width)
        boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, height)

        keep_indices = batched_nms(boxes_tensor, scores_tensor, classes_tensor, self.nms_iou)

        boxes_np = boxes_tensor[keep_indices].numpy()
        classes_np = classes_tensor[keep_indices].numpy()
        scores_np = scores_tensor[keep_indices].numpy()

        return boxes_np, classes_np, scores_np
