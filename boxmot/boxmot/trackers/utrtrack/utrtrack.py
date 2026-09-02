# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from torch import device

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.utrtrack.sort.detection import Detection
from boxmot.trackers.utrtrack.sort.tracker import TrackerTLUKF
from boxmot.trackers.utrtrack.sort.linear_assignment import NearestNeighborDistanceMetric_TLUKF
from boxmot.utils.ops import xyxy2tlwh

from boxmot.utils import logger as LOGGER


class UTRTrack(BaseTracker):
    """
    UTRTrack: a StrongSORT-style tracker built around a dual-tracker Transfer
    Learning Unscented Kalman Filter (TL-UKF) with Virtual Trajectory
    Augmentation for endoscopy videos.

    Additional Features:
    - Virtual trajectory prediction for lost tracks via transfer learning
      (Source -> Primary), gated by recent high-quality evidence
    - Appearance-based track recovery (single_object_mode)
    - Adaptive/decaying confidence for virtual detections

    Additional Parameters (beyond BaseTracker / StrongSORT-style ones):
    - single_object_mode (bool): prioritize re-associating the one tracked
      object to its existing ID instead of spawning a new one (default: False)
    - reacquire_min_iou / reacquire_max_cost / reacquire_max_age: thresholds
      used by single_object_mode's re-association step
    - max_virtual_age (int): maximum consecutive missed frames a track may
      still emit a virtual (predicted) box for (default: 30)
    - virtual_recent_hq_gap (int): how many frames a virtual box is allowed
      to lag behind the Source tracker's last high-quality update (default: 6)
    - virtual_conf (float): base confidence assigned to virtual detections,
      decayed geometrically with miss duration (default: 0.3)
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        # BaseTracker parameters
        det_thresh: float = 0.3,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        # StrongSort-style parameters
        min_conf: float = 0.1,
        max_cos_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
        # UTRTrack-specific parameters
        single_object_mode: bool = False,
        reacquire_min_iou: float = 0.05,
        reacquire_max_cost: float = 0.55,
        reacquire_max_age: int = 20,
        max_virtual_age: int = 30,
        virtual_recent_hq_gap: int = 6,
        virtual_conf: float = 0.3,
        **kwargs  # Additional BaseTracker parameters
    ):
        # Forward all BaseTracker parameters explicitly
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            max_obs=max_obs,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            per_class=per_class,
            nr_classes=nr_classes,
            asso_func=asso_func,
            is_obb=is_obb,
            **kwargs
        )

        self.min_conf = min_conf

        # Initialize ReID model
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        # Initialize the TL-UKF multi-target tracker
        self.tracker = TrackerTLUKF(
            metric=NearestNeighborDistanceMetric_TLUKF("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            single_object_mode=single_object_mode,
            reacquire_min_iou=reacquire_min_iou,
            reacquire_max_cost=reacquire_max_cost,
            reacquire_max_age=reacquire_max_age,
        )

        # Initialize camera motion compensation
        self.cmc = get_cmc_method("ecc")()
        # Confidence to assign virtual (missed) boxes in outputs
        self.virtual_conf: float = virtual_conf
        self.max_virtual_age = max_virtual_age
        self.virtual_recent_hq_gap = virtual_recent_hq_gap
        self.frame_idx = 0

        LOGGER.success("Initialized UTRTrack")

    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
        if embs is not None:
            assert (
                dets.shape[0] == embs.shape[0]
            ), "Missmatch between detections and embeddings sizes"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        remain_inds = dets[:, 4] >= self.min_conf
        dets = dets[remain_inds]

        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs[remain_inds]
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(
                tlwh, confs, clss, det_ind, features
            )
        ]

        self.frame_idx += 1

        # update tracker with frame dimensions for boundary checking
        self.tracker.predict()
        img_height, img_width = img.shape[:2]
        self.tracker.update(
            detections,
            frame_id=self.frame_idx,
            img_width=img_width,
            img_height=img_height,
        )

        # output bbox identities
        outputs = []

        # Single loop to output ONLY ONE box per track: prevents duplicate
        # outputs (both real and virtual) for the same track.
        #
        # Logic:
        # - time_since_update == 0: Track matched -> output REAL box
        # - time_since_update >= 1: Track unmatched -> output VIRTUAL box
        # Each track enters EXACTLY ONE branch - no duplicates possible

        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue

            if track.time_since_update == 0:
                # MATCHED TRACK: Output real detection with original confidence
                x1, y1, x2, y2 = track.to_tlbr()
                id = track.id
                conf = track.conf  # Real confidence from detection
                cls = track.cls
                det_ind = track.det_ind

                outputs.append(
                    np.concatenate(
                        ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                    ).reshape(1, -1)
                )
            elif track.time_since_update >= 1:
                # UNMATCHED TRACK: Output virtual box with predicted position.
                # Transfer learning has already been applied in TrackerTLUKF.update()
                if track.time_since_update > self.max_virtual_age:
                    continue

                # Only output a virtual box when transfer learning was truly applied.
                if not getattr(track, "transfer_active", False):
                    continue

                # Require recent high-quality evidence to avoid ghost boxes.
                last_hq = getattr(track, "last_high_quality_frame", -10**9)
                if (self.frame_idx - last_hq) > self.virtual_recent_hq_gap:
                    continue

                x1, y1, x2, y2 = track.to_tlbr()

                # Validate virtual box dimensions before output
                if x2 <= x1 or y2 <= y1:
                    continue

                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # Skip degenerate or tiny boxes
                if box_area < 100:  # Minimum 100 pixels
                    continue

                # Skip boxes with unreasonable aspect ratio
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                    continue

                id = track.id
                # Decay virtual confidence as miss duration grows.
                conf = max(0.12, self.virtual_conf * (0.85 ** (track.time_since_update - 1)))
                cls = track.cls
                det_ind = getattr(track, 'det_ind', 0)

                outputs.append(
                    np.concatenate(
                        ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                    ).reshape(1, -1)
                )

        if len(outputs) > 0:
            outputs = np.concatenate(outputs)

            # Sanity check: duplicate IDs in the same frame should never happen
            # with the single-loop-per-track logic above.
            unique_ids = set()
            duplicate_ids = []
            for output in outputs:
                track_id = int(output[4])
                if track_id in unique_ids:
                    duplicate_ids.append(track_id)
                else:
                    unique_ids.add(track_id)

            if duplicate_ids:
                import warnings
                warnings.warn(f"CRITICAL: Duplicate track IDs in same frame: {duplicate_ids}")

            # Apply NMS to remove overlapping boxes (belt-and-suspenders approach)
            outputs = self._apply_nms(outputs, iou_threshold=0.5)
            return outputs
        return np.array([])

    def _apply_nms(self, tracks, iou_threshold=0.5):
        """
        Apply NMS to remove overlapping boxes:
        1. Same ID: keep only one box - priority real > virtual, then highest conf
        2. Different IDs but overlapping: suppress virtual if a real box exists
        3. At most one virtual box is kept per frame

        Args:
            tracks: numpy array [x1, y1, x2, y2, id, conf, cls, det_ind]
            iou_threshold: IoU threshold for considering boxes as overlapping

        Returns:
            Filtered tracks after NMS
        """
        if len(tracks) == 0:
            return tracks

        # Step 1: Group by ID and keep only the best box per ID
        id_to_tracks = {}
        for i, track in enumerate(tracks):
            track_id = int(track[4])
            id_to_tracks.setdefault(track_id, []).append(track)

        # Step 2: For each ID, select the best box
        filtered_tracks = []
        for track_id, track_group in id_to_tracks.items():
            if len(track_group) == 1:
                filtered_tracks.append(track_group[0])
                continue

            real_boxes = [t for t in track_group if t[5] >= 0.35]  # Real detections
            virtual_boxes = [t for t in track_group if t[5] < 0.35]  # Virtual predictions

            if len(real_boxes) > 0:
                best_real = max(real_boxes, key=lambda t: t[5])
                filtered_tracks.append(best_real)
            elif len(virtual_boxes) > 0:
                filtered_tracks.append(virtual_boxes[0])

        # Step 3: Spatial NMS across different IDs + limit virtual boxes per frame
        if len(filtered_tracks) == 0:
            return np.array([])

        filtered_tracks = np.array(filtered_tracks)
        sorted_indices = np.argsort(-filtered_tracks[:, 5])  # Sort by conf descending
        sorted_tracks = filtered_tracks[sorted_indices]

        keep = []
        virtual_count = 0
        MAX_VIRTUAL_PER_FRAME = 1

        for i, track in enumerate(sorted_tracks):
            track_id = int(track[4])
            track_conf = track[5]
            track_box = track[:4]
            is_virtual = track_conf < 0.35

            if is_virtual and virtual_count >= MAX_VIRTUAL_PER_FRAME:
                continue

            should_keep = True
            for kept_idx in keep:
                kept_track = sorted_tracks[kept_idx]
                kept_id = int(kept_track[4])
                kept_conf = kept_track[5]
                kept_box = kept_track[:4]

                if kept_id == track_id:
                    # Same ID (shouldn't happen after Step 2, but be safe)
                    should_keep = False
                    break

                iou = self._calculate_iou(track_box, kept_box)

                if iou > iou_threshold:
                    if track_conf < 0.35 and kept_conf >= 0.35:
                        # Current is virtual, kept is real - suppress virtual
                        should_keep = False
                        break

            if should_keep:
                keep.append(i)
                if is_virtual:
                    virtual_count += 1

        return sorted_tracks[keep]

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def reset(self):
        pass
