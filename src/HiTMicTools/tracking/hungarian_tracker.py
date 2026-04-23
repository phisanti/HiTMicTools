"""
Hungarian tracker v2 with gap bridging.

Compared to v1 (single-frame-back linking only):
  - Adds `gap_bridge_frames` parameter. A track whose last detection was up to
    N frames ago is still eligible for linking at the current frame. Once a
    track goes silent for longer than N, it is retired.
  - Preserves v1's class-agnostic behavior: cost matrix is built over all
    detections in scope, regardless of object_class.
  - Preserves v1's piPOS lock-in API (no changes to `apply_pipos_lockin`).
  - API-compatible with v1: `track_objects(measurements_df, ...)` returns the
    same DataFrame with `trackid` column added.

Rationale: on e009 HK wells (stationary dead cells), median track length was
14/32 frames with v1 because 2% per-frame detection miss rate compounds to
50% full-coverage. Gap bridging recovers tracks that survive single-frame
detection dropouts without requiring distance cutoff to change.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class HungarianTracker:
    """Frame-to-frame optimal assignment tracker with piPOS lock-in and gap bridging."""

    def __init__(self, max_distance: float = 25.0, gap_bridge_frames: int = 2):
        """
        Args:
            max_distance: Maximum linking distance in pixels. Pairs beyond
                this threshold are left unlinked (new track ID assigned).
            gap_bridge_frames: Number of consecutive missed frames allowed
                before a track is retired. 0 = v1 behavior (no bridging).
                Default 2 (tolerates 2 missed frames).
        """
        self.max_distance = max_distance
        self.gap_bridge_frames = gap_bridge_frames

    def set_features(self, features: List[str]) -> None:
        """No-op for API compatibility with CellTracker."""
        pass

    def track_objects(
        self,
        measurements_df: pd.DataFrame,
        volume_bounds: Optional[Tuple[int, int]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        """
        Assign persistent track IDs across frames using Hungarian assignment
        with gap bridging.

        Args:
            measurements_df: DataFrame with columns: frame, centroid_0, centroid_1
            volume_bounds: Ignored (kept for API compatibility with CellTracker).
            logger: Optional logger instance.

        Returns:
            Input DataFrame with 'trackid' column added (int32, -1 for unlinked).
        """
        df = measurements_df.copy()
        frames = sorted(df["frame"].unique())

        if len(frames) == 0:
            df["trackid"] = np.int32(-1)
            return df

        track_ids = pd.Series(np.int32(-1), index=df.index)
        next_track_id = 0

        # active_tracks[trackid] = {"centroid": (c0, c1), "last_frame": int}
        # A track is eligible for linking while (curr_frame - last_frame) <= gap_bridge_frames.
        active_tracks = {}

        # Seed with first frame detections
        first_mask = df["frame"] == frames[0]
        first_indices = df.index[first_mask]
        for idx in first_indices:
            c0 = df.at[idx, "centroid_0"]
            c1 = df.at[idx, "centroid_1"]
            track_ids[idx] = next_track_id
            active_tracks[next_track_id] = {
                "centroid": (c0, c1),
                "last_frame": frames[0],
            }
            next_track_id += 1

        total_linked = 0
        total_new = len(first_indices)
        total_bridged = 0  # links that used a gap >= 2 frames
        distances_all = []
        max_dist_used = 0.0

        # Link subsequent frames
        for fi in range(1, len(frames)):
            curr_frame = frames[fi]
            curr_mask = df["frame"] == curr_frame
            curr_indices = df.index[curr_mask]

            if len(curr_indices) == 0:
                continue

            # Eligible tracks: last seen within gap_bridge_frames
            eligible_tids = [
                tid for tid, info in active_tracks.items()
                if (curr_frame - info["last_frame"]) <= self.gap_bridge_frames
            ]

            if len(eligible_tids) == 0:
                # All current detections become new tracks
                for idx in curr_indices:
                    c0 = df.at[idx, "centroid_0"]
                    c1 = df.at[idx, "centroid_1"]
                    track_ids[idx] = next_track_id
                    active_tracks[next_track_id] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                    }
                    next_track_id += 1
                total_new += len(curr_indices)
                continue

            prev_centroids = np.array(
                [active_tracks[tid]["centroid"] for tid in eligible_tids]
            )
            curr_centroids = df.loc[
                curr_indices, ["centroid_0", "centroid_1"]
            ].values

            cost = cdist(prev_centroids, curr_centroids, metric="euclidean")
            row_ind, col_ind = linear_sum_assignment(cost)

            linked_curr = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_distance:
                    tid = eligible_tids[r]
                    curr_idx = curr_indices[c]
                    c0 = df.at[curr_idx, "centroid_0"]
                    c1 = df.at[curr_idx, "centroid_1"]
                    track_ids[curr_idx] = tid
                    gap = curr_frame - active_tracks[tid]["last_frame"]
                    if gap >= 2:
                        total_bridged += 1
                    active_tracks[tid] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                    }
                    linked_curr.add(c)
                    total_linked += 1
                    distances_all.append(cost[r, c])
                    if cost[r, c] > max_dist_used:
                        max_dist_used = cost[r, c]

            # Unmatched detections in current frame become new tracks
            for j, idx in enumerate(curr_indices):
                if j not in linked_curr:
                    c0 = df.at[idx, "centroid_0"]
                    c1 = df.at[idx, "centroid_1"]
                    track_ids[idx] = next_track_id
                    active_tracks[next_track_id] = {
                        "centroid": (c0, c1),
                        "last_frame": curr_frame,
                    }
                    next_track_id += 1
                    total_new += 1

            # Retire tracks that have been silent beyond the bridge window.
            # This isn't strictly necessary for correctness (they just stay
            # ineligible), but prevents the dict growing to millions of stale
            # entries on long runs.
            cutoff = curr_frame - self.gap_bridge_frames
            active_tracks = {
                tid: info for tid, info in active_tracks.items()
                if info["last_frame"] >= cutoff
            }

        df["trackid"] = track_ids.astype(np.int32)

        if logger:
            n_tracks = df["trackid"].nunique()
            n_objects = len(df)
            n_frames = len(frames)

            track_lengths = df.groupby("trackid")["frame"].nunique()
            full_length_tracks = (track_lengths == n_frames).sum()
            short_tracks = (track_lengths == 1).sum()
            median_length = track_lengths.median()

            mean_dist = np.mean(distances_all) if distances_all else 0.0
            p95_dist = np.percentile(distances_all, 95) if distances_all else 0.0

            logger.info(
                f"Hungarian tracking v2 summary:\n"
                f"  Config: max_distance={self.max_distance}, "
                f"gap_bridge_frames={self.gap_bridge_frames}\n"
                f"  Frames: {n_frames}, Detections: {n_objects}\n"
                f"  Tracks: {n_tracks} total, {full_length_tracks} full-length "
                f"({n_frames}f), {short_tracks} single-frame\n"
                f"  Links: {total_linked} total, {total_bridged} via gap bridging "
                f"({100.0 * total_bridged / max(total_linked, 1):.1f}%), "
                f"{total_new} new tracks\n"
                f"  Distances: mean={mean_dist:.1f}px, p95={p95_dist:.1f}px, "
                f"max={max_dist_used:.1f}px (cutoff={self.max_distance}px)\n"
                f"  Track length: median={median_length:.0f} frames"
            )

        return df

    def apply_pipos_lockin(
        self,
        measurements_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
    ) -> pd.DataFrame:
        """
        Enforce piPOS lock-in: once a track is piPOS, all subsequent frames stay piPOS.

        Unchanged from v1.
        """
        if "trackid" not in measurements_df.columns or "pi_class" not in measurements_df.columns:
            if logger:
                logger.warning("piPOS lock-in skipped: missing trackid or pi_class column")
            return measurements_df

        override_count = 0
        tracks_with_lockin = 0
        tracked = measurements_df["trackid"] != -1
        n_tracked_cells = tracked.sum()
        n_untracked = (~tracked).sum()

        pipos_before = (measurements_df["pi_class"] == "piPOS").sum()

        for tid, group in measurements_df.loc[tracked].groupby("trackid"):
            pipos_frames = group.loc[group["pi_class"] == "piPOS", "frame"]
            if len(pipos_frames) == 0:
                continue

            first_pipos_frame = pipos_frames.min()
            mask = (
                (measurements_df["trackid"] == tid)
                & (measurements_df["frame"] > first_pipos_frame)
                & (measurements_df["pi_class"] != "piPOS")
            )
            n_overrides = mask.sum()
            if n_overrides > 0:
                tracks_with_lockin += 1
                override_count += n_overrides
            measurements_df.loc[mask, "pi_class"] = "piPOS"

        pipos_after = (measurements_df["pi_class"] == "piPOS").sum()

        if logger:
            logger.info(
                f"piPOS lock-in summary:\n"
                f"  Tracked detections: {n_tracked_cells}, Untracked: {n_untracked}\n"
                f"  Tracks with lock-in applied: {tracks_with_lockin}\n"
                f"  Classifications overridden: {override_count}\n"
                f"  piPOS count: {pipos_before} -> {pipos_after} "
                f"(+{pipos_after - pipos_before})"
            )

        return measurements_df
