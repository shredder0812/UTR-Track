# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
"""UTR-Track-v1: public tracker class."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from torch import device

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.utrtrack.sort.detection import Detection
from boxmot.trackers.utrtrack.sort.linear_assignment import NearestNeighborDistanceMetric_TLUKF
from boxmot.trackers.utrtrack_v1.sort.tracker import TrackerV1
from boxmot.utils import logger as LOGGER
from boxmot.utils.ops import xyxy2tlwh


class UTRTrackV1(BaseTracker):
    """UTR-Track-v1: StrongSORT-style lesion tracker for endoscopy video.

    Builds on UTR-Track-v0 (dual Source/Primary TL-UKF with virtual trajectory
    augmentation, OSNet-DCN appearance, ECC camera-motion compensation) and changes:

    * XYSR state ``[x, y, area, w/h]`` with a **calibrated process noise** ``Q``
      (``q_calibration_scale``);
    * **three-tier association**: appearance cascade, IoU, then low-confidence
      detections (``conf < byte_low_conf_thresh``) for confirmed tracks only;
    * track confirmation after ``n_init = 3`` matches;
    * Hungarian **re-acquisition** of lost tracks (``reacquire_min_iou``, ``reacquire_max_cost``);
    * **identity-preserving birth control** (``birth_iou``, ``birth_contain``):
      a new detection never replaces a live track;
    * **lost-track absorption** (``absorption_iou``, ``absorption_app_sim``).

    It has no single-object mode and no cap on the number of tracks.
    Class-agnostic NMS of the detections (IoU 0.45) is part of the v1 configuration
    but is applied by the caller before :meth:`update` (see ``utrtrack_pipeline.py``).

    The defaults below are the configuration reported as UTR-Track-v1
    (IGH, 18 sequences: IDF1 85.4, MOTA 62.4, IDs 0).

    Args:
        reid_weights: path to the ReID weights (v1 uses ``osnet_dcn_x0_5_hybrid.pt``).
        device: torch device of the ReID model.
        half: run the ReID model in fp16.
        min_conf: detections below this confidence are discarded on input.
        max_cos_dist: appearance gate of the cascade (cosine distance).
        max_iou_dist: maximum ``1 - IoU`` of the IoU tiers.
        max_age: miss-counter limit before deletion (two steps per missed frame).
        n_init: matches needed to confirm a track.
        nn_budget: size of the per-track appearance gallery in the metric.
        mc_lambda: weight of the appearance cost vs the Mahalanobis distance.
        ema_alpha: EMA factor of the track appearance features.
        reacquire_min_iou: minimum IoU for re-acquiring a lost track.
        reacquire_max_cost: maximum appearance distance for re-acquiring a lost track.
        reacquire_max_age: lost tracks older than this (miss counter) are not re-acquired;
            ``None`` means ``max_age``.
        max_virtual_age: virtual boxes are emitted while the miss counter is ``<=`` this.
        virtual_recent_hq_gap: virtual boxes (and transfer learning) require a
            high-confidence update within this many frames.
        virtual_conf: score of the first virtual box; it decays by 0.85 per miss step.
        byte_low_conf_thresh: detections below this go only to the low-confidence tier.
        byte_max_iou_dist: ``1 - IoU`` limit of the low-confidence tier (``None`` = ``max_iou_dist``).
        q_calibration_scale: process-noise calibration factor ``m`` (``None`` = hand-set ``Q``).
        absorption_iou / absorption_app_sim: thresholds of lost-track absorption.
        birth_iou / birth_contain: overlap and containment thresholds of birth control
            (``birth_contain=None`` disables the containment test).
        max_virtual_per_frame: low-score (< 0.35) boxes kept per frame by the output NMS.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        # BaseTracker parameters
        det_thresh: float = 0.3,
        max_age: int = 500,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        # StrongSORT-style parameters
        min_conf: float = 0.1,
        max_cos_dist: float = 0.9,
        max_iou_dist: float = 0.9,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.95,
        # UTR-Track-v1 parameters
        reacquire_min_iou: float = 0.01,
        reacquire_max_cost: float = 0.75,
        reacquire_max_age: int | None = None,
        max_virtual_age: int = 5,
        virtual_recent_hq_gap: int = 4,
        virtual_conf: float = 0.28,
        byte_low_conf_thresh: float = 0.45,
        byte_max_iou_dist: float | None = None,
        q_calibration_scale: float | None = 1.0,
        absorption_iou: float = 0.3,
        absorption_app_sim: float = 0.7,
        birth_iou: float = 0.3,
        birth_contain: float | None = 0.8,
        max_virtual_per_frame: int = 1,
        **kwargs,
    ):
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
            **kwargs,
        )
        self.min_conf = min_conf
        self.model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model

        self.tracker = TrackerV1(
            metric=NearestNeighborDistanceMetric_TLUKF("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            ema_alpha=ema_alpha,
            mc_lambda=mc_lambda,
            reacquire_min_iou=reacquire_min_iou,
            reacquire_max_cost=reacquire_max_cost,
            reacquire_max_age=max_age if reacquire_max_age is None else reacquire_max_age,
            # The window that gates virtual-box output also gates transfer learning.
            transfer_stale_gap=virtual_recent_hq_gap,
            byte_low_conf_thresh=byte_low_conf_thresh,
            byte_max_iou_dist=byte_max_iou_dist,
            q_calibration_scale=q_calibration_scale,
            absorption_iou=absorption_iou,
            absorption_app_sim=absorption_app_sim,
            birth_iou=birth_iou,
            birth_contain=birth_contain,
        )

        self.cmc = get_cmc_method("ecc")()
        self.virtual_conf = virtual_conf
        self.max_virtual_age = max_virtual_age
        self.virtual_recent_hq_gap = virtual_recent_hq_gap
        self.max_virtual_per_frame = max_virtual_per_frame
        self.frame_idx = 0

        LOGGER.success("Initialized UTR-Track-v1")

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """Track one frame.

        Args:
            dets: ``(N, 6)`` detections ``[x1, y1, x2, y2, conf, cls]``.
            img: BGR frame (used for ReID crops and camera-motion compensation).
            embs: optional precomputed ``(N, D)`` appearance embeddings.

        Returns:
            ``(M, 8)`` array ``[x1, y1, x2, y2, id, conf, cls, det_ind]`` with at most one
            box per confirmed track: a real box when matched in this frame, otherwise a
            virtual box (score ``< 0.35``) while the virtual-box conditions hold.
        """
        assert isinstance(dets, np.ndarray), f"Unsupported 'dets' input format '{type(dets)}', expected np.ndarray"
        assert isinstance(img, np.ndarray), f"Unsupported 'img' input format '{type(img)}', expected np.ndarray"
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
        if embs is not None:
            assert dets.shape[0] == embs.shape[0], "Missmatch between detections and embeddings sizes"

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

        if embs is not None:
            features = embs[remain_inds]
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(tlwh, confs, clss, det_ind, features)
        ]

        self.frame_idx += 1
        self.tracker.predict()
        img_height, img_width = img.shape[:2]
        self.tracker.update(detections, frame_id=self.frame_idx, img_width=img_width, img_height=img_height)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update == 0:
                x1, y1, x2, y2 = track.to_tlbr()
                outputs.append(np.concatenate(
                    ([x1, y1, x2, y2], [track.id], [track.conf], [track.cls], [track.det_ind])
                ).reshape(1, -1))
            elif track.time_since_update >= 1:
                box = self._virtual_box(track)
                if box is not None:
                    outputs.append(box)

        if len(outputs) == 0:
            return np.array([])
        outputs = np.concatenate(outputs)
        ids = [int(o[4]) for o in outputs]
        if len(ids) != len(set(ids)):
            warnings.warn(f"Duplicate track IDs in the same frame: {ids}")
        return self._apply_nms(outputs, iou_threshold=0.5)

    def _virtual_box(self, track) -> np.ndarray | None:
        """Virtual (VTA) box of an unmatched confirmed track, or ``None`` if not allowed.

        Emitted only while the miss counter is ``<= max_virtual_age``, transfer learning
        ran in this frame, the last high-confidence update is at most
        ``virtual_recent_hq_gap`` frames old, and the box is not degenerate
        (area >= 100 px^2, aspect ratio in [0.1, 10]).
        """
        if track.time_since_update > self.max_virtual_age:
            return None
        if not track.transfer_active:
            return None
        if (self.frame_idx - track.last_high_quality_frame) > self.virtual_recent_hq_gap:
            return None
        x1, y1, x2, y2 = track.to_tlbr()
        if x2 <= x1 or y2 <= y1:
            return None
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width * box_height < 100:
            return None
        aspect_ratio = box_width / box_height if box_height > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            return None
        conf = max(0.12, self.virtual_conf * (0.85 ** (track.time_since_update - 1)))
        return np.concatenate(
            ([x1, y1, x2, y2], [track.id], [conf], [track.cls], [track.det_ind])
        ).reshape(1, -1)

    def _apply_nms(self, tracks: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Output filtering.

        1. One box per id: the highest-scoring real box (score >= 0.35) if any, else a virtual box.
        2. Boxes are visited by decreasing score; a low-score box (< 0.35, i.e. virtual or a
           weak tier-C box) is dropped if it overlaps (IoU > ``iou_threshold``) a kept
           real box of another id, and at most ``max_virtual_per_frame`` low-score boxes are kept.
        """
        if len(tracks) == 0:
            return tracks

        id_to_tracks = {}
        for track in tracks:
            id_to_tracks.setdefault(int(track[4]), []).append(track)

        filtered_tracks = []
        for track_group in id_to_tracks.values():
            if len(track_group) == 1:
                filtered_tracks.append(track_group[0])
                continue
            real_boxes = [t for t in track_group if t[5] >= 0.35]
            virtual_boxes = [t for t in track_group if t[5] < 0.35]
            if len(real_boxes) > 0:
                filtered_tracks.append(max(real_boxes, key=lambda t: t[5]))
            elif len(virtual_boxes) > 0:
                filtered_tracks.append(virtual_boxes[0])

        if len(filtered_tracks) == 0:
            return np.array([])

        filtered_tracks = np.array(filtered_tracks)
        sorted_tracks = filtered_tracks[np.argsort(-filtered_tracks[:, 5])]

        keep = []
        virtual_count = 0
        for i, track in enumerate(sorted_tracks):
            track_id = int(track[4])
            track_conf = track[5]
            is_virtual = track_conf < 0.35
            if is_virtual and virtual_count >= self.max_virtual_per_frame:
                continue
            should_keep = True
            for kept_idx in keep:
                kept_track = sorted_tracks[kept_idx]
                if int(kept_track[4]) == track_id:
                    should_keep = False
                    break
                if self._calculate_iou(track[:4], kept_track[:4]) > iou_threshold:
                    if track_conf < 0.35 and kept_track[5] >= 0.35:
                        should_keep = False
                        break
            if should_keep:
                keep.append(i)
                if is_virtual:
                    virtual_count += 1
        return sorted_tracks[keep]

    @staticmethod
    def _calculate_iou(box1, box2) -> float:
        """IoU of two ``[x1, y1, x2, y2]`` boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def reset(self) -> None:
        """No state to reset beyond construction."""
        pass
