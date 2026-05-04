import numpy as np
import pandas as pd

from HiTMicTools.img_processing.mask_ops import (
    map_predictions_to_labels,
    map_predictions_to_labels_by_frame,
)


def test_map_predictions_to_labels_by_frame_keeps_reused_label_ids_separate():
    labeled_stack = np.array(
        [
            [[1, 1, 0], [2, 2, 0]],
            [[1, 1, 0], [2, 2, 0]],
        ]
    )
    measurements = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "label": [1, 2, 1, 2],
            "object_class": ["single-cell", "clump", "clump", "single-cell"],
        }
    )

    mapped = map_predictions_to_labels_by_frame(
        labeled_stack,
        measurements,
        "object_class",
        value_map={"single-cell": 1, "clump": 2},
    )

    expected = np.array(
        [
            [[1, 1, 0], [2, 2, 0]],
            [[2, 2, 0], [1, 1, 0]],
        ]
    )
    np.testing.assert_array_equal(mapped, expected)


def test_global_mapping_reproduces_last_frame_overwrite_for_reused_label_ids():
    labeled_stack = np.array(
        [
            [[1, 1, 0], [2, 2, 0]],
            [[1, 1, 0], [2, 2, 0]],
        ]
    )

    mapped = map_predictions_to_labels(
        labeled_stack,
        ["single-cell", "clump", "clump", "single-cell"],
        [1, 2, 1, 2],
        value_map={"single-cell": 1, "clump": 2},
    )

    expected_last_frame_assignment = np.array(
        [
            [[2, 2, 0], [1, 1, 0]],
            [[2, 2, 0], [1, 1, 0]],
        ]
    )
    np.testing.assert_array_equal(mapped, expected_last_frame_assignment)
