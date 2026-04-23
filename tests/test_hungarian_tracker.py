"""Quick verification test for HungarianTracker."""

import numpy as np
import pandas as pd
from HiTMicTools.tracking.hungarian_tracker import HungarianTracker


def test_tracking_and_lockin():
    # 5 cells, 3 frames, slight movement (2px jitter)
    np.random.seed(42)
    records = []
    base_positions = np.array(
        [[100, 100], [200, 200], [300, 300], [400, 400], [500, 500]]
    )
    for frame in range(3):
        for i, (y, x) in enumerate(base_positions):
            noise = np.random.randn(2) * 2
            records.append(
                {
                    "frame": frame,
                    "label": i + 1,
                    "centroid_0": y + noise[0],
                    "centroid_1": x + noise[1],
                    "pi_class": "piNEG",
                }
            )

    df = pd.DataFrame(records)

    # Cell 2 dies in frame 1, flickers back to piNEG in frame 2
    df.loc[(df["frame"] == 1) & (df["label"] == 2), "pi_class"] = "piPOS"
    df.loc[(df["frame"] == 2) & (df["label"] == 2), "pi_class"] = "piNEG"

    tracker = HungarianTracker(max_distance=25.0)
    df = tracker.track_objects(df)

    # All 5 cells should get consistent track IDs across frames
    print("Track IDs per frame:")
    for f in range(3):
        ids = df.loc[df["frame"] == f, "trackid"].tolist()
        print(f"  Frame {f}: {ids}")

    # Each cell should have exactly 1 unique trackid across all 3 frames
    for label in range(1, 6):
        track_ids = df.loc[df["label"] == label, "trackid"].unique()
        assert len(track_ids) == 1, f"Cell {label} has multiple track IDs: {track_ids}"
    print("PASS: All cells have consistent track IDs")

    # Test lock-in
    df = tracker.apply_pipos_lockin(df)
    cell2_track = df.loc[(df["frame"] == 0) & (df["label"] == 2), "trackid"].iloc[0]
    cell2_classes = df.loc[df["trackid"] == cell2_track, "pi_class"].tolist()
    assert cell2_classes == [
        "piNEG",
        "piPOS",
        "piPOS",
    ], f"Lock-in failed: {cell2_classes}"
    print(f"PASS: Cell 2 lock-in correct: {cell2_classes}")

    # Other cells should remain piNEG
    for label in [1, 3, 4, 5]:
        tid = df.loc[(df["frame"] == 0) & (df["label"] == label), "trackid"].iloc[0]
        classes = df.loc[df["trackid"] == tid, "pi_class"].tolist()
        assert all(c == "piNEG" for c in classes), f"Cell {label} should be all piNEG"
    print("PASS: Other cells remain piNEG")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_tracking_and_lockin()
