# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
"""Track object of UTR-Track-v1 (dual Source/Primary TL-UKF in XYSR space)."""

from __future__ import annotations

import numpy as np

from boxmot.motion.kalman_filters.aabb.xysr_tlukf_v1 import KalmanFilterXYSRTLUKFv1
from boxmot.trackers.utrtrack.sort.track import _build_3x3_warp, _ensure_scalar


class TrackV1:
    """One lesion trajectory of UTR-Track-v1.

    Each track owns two filters initialised from the same detection:

    * **Source** (teacher): updated only by detections with ``conf >= high_conf_threshold``.
    * **Primary** (student): updated by every matched detection. When the track is
      unmatched, Primary is updated with the Source prediction (transfer learning), and
      the resulting state is what a virtual box (VTA) is drawn from.

    The miss counter ``time_since_update`` grows by **two** per missed frame (once in
    :meth:`predict`, once in :meth:`apply_transfer_learning`). Every threshold expressed
    in it (``max_age``, ``max_virtual_age`` of the tracker) is therefore in update steps,
    not frames: ``max_age=500`` keeps a lost track for about 250 frames, and
    ``max_virtual_age=5`` allows at most two virtual boxes per gap.

    Args:
        detection: :class:`~boxmot.trackers.utrtrack.sort.detection.Detection` that starts the track.
        id: track identity.
        n_init: number of matches (including the first) needed to confirm the track.
        max_age: miss-counter value after which the track is deleted.
        ema_alpha: EMA factor of the appearance feature gallery.
        high_conf_threshold: minimum confidence for a detection to update the Source filter.
        transfer_stale_gap: frames since the last high-confidence update after which
            transfer learning is skipped and the Primary velocity is set to zero.
        q_calibration_scale: process-noise calibration factor passed to both filters.
    """

    def __init__(
        self,
        detection,
        id: int,
        n_init: int,
        max_age: int,
        ema_alpha: float,
        high_conf_threshold: float = 0.6,
        transfer_stale_gap: int = 4,
        q_calibration_scale: float | None = 1.0,
    ):
        self.id = id
        bbox = detection.to_xysr()
        self.bbox = bbox
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.state = 1  # 1 = tentative, 2 = confirmed
        self.features = []

        # Majority vote over the last 200 class labels keeps the label stable when the
        # detector flips classes between frames (per_class=False tracking).
        self.class_history = [self.cls]
        self.class_vote_window = 200
        self.stable_cls = self.cls
        if detection.feat is not None:
            feat = detection.feat / (np.linalg.norm(detection.feat) + 1e-6)
            self.features = [feat]
        self._n_init = n_init
        self._max_age = max_age
        self.high_conf_threshold = high_conf_threshold
        self.transfer_stale_gap = transfer_stale_gap

        self.source_kf = KalmanFilterXYSRTLUKFv1(is_source=True, q_calibration_scale=q_calibration_scale)
        self.primary_kf = KalmanFilterXYSRTLUKFv1(is_source=False, q_calibration_scale=q_calibration_scale)
        self.source_kf.initiate(self.bbox)
        self.primary_kf.initiate(self.bbox)

        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

        # Virtual trajectory bookkeeping (kept for post-hoc analysis and figures)
        self.virtual_boxes = []
        self.last_high_quality_frame = 0
        self.last_real_frame = 0
        self.transfer_active = False

        # Static-scene handling (e.g. a paused video): stop drifting after 3 still frames
        self.last_position = bbox[:2].copy()
        self.static_frame_count = 0
        self.position_threshold = 1.0  # pixels

    # ------------------------------------------------------------- box access
    def to_tlwh(self) -> np.ndarray:
        """Current Primary state as ``[top-left x, top-left y, w, h]``."""
        x, y, s, r = self.kf.x[:4]
        s = max(float(s), 0.0)
        r = max(float(r), 1e-6)
        w = np.sqrt(s * r)
        h = np.sqrt(s / r) if r > 0 else 0.0
        return np.array([x - w / 2, y - h / 2, w, h], dtype=np.float32)

    def to_tlbr(self) -> np.ndarray:
        """Current Primary state as ``[x1, y1, x2, y2]``."""
        x1, y1, w, h = self.to_tlwh()
        return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

    # ------------------------------------------------------------- life cycle
    def predict(self) -> None:
        """Predict both filters one frame ahead and freeze the track on a static scene."""
        self.source_kf.predict()
        self.primary_kf.predict()

        current_pos = self.primary_kf.x[:2].copy()
        pos_change = np.linalg.norm(current_pos - self.last_position)
        if pos_change < self.position_threshold:
            self.static_frame_count += 1
            if self.static_frame_count >= 3:
                self.source_kf.x[:2] = self.last_position.copy()
                self.primary_kf.x[:2] = self.last_position.copy()
                self.source_kf.x[4:8] = 0.0
                self.primary_kf.x[4:8] = 0.0
        else:
            self.static_frame_count = 0
            self.last_position = current_pos.copy()

        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, frame_id: int | None = None) -> None:
        """Update the track with a matched detection.

        Primary always takes the measurement; Source only when the detection confidence
        reaches ``high_conf_threshold`` (which also refreshes ``last_high_quality_frame``).
        """
        bbox = detection.to_xysr()
        conf = detection.conf
        self.static_frame_count = 0

        self.primary_kf.update(measurement=bbox, confidence=conf)
        self.transfer_active = False
        if frame_id is not None:
            self.last_real_frame = frame_id

        if conf >= self.high_conf_threshold:
            self.source_kf.update(measurement=bbox, confidence=conf)
            if frame_id is not None:
                self.last_high_quality_frame = frame_id

        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()
        self.bbox = bbox
        self.last_position = self.primary_kf.x[:2].copy()
        self.hits += 1
        self.time_since_update = 0
        self.conf = conf

        # Class voting
        self.class_history.append(detection.cls)
        if len(self.class_history) > self.class_vote_window:
            self.class_history.pop(0)
        class_counts = {}
        for cls in self.class_history:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        self.stable_cls = max(class_counts, key=class_counts.get)
        self.cls = self.stable_cls
        self.det_ind = detection.det_ind

        # Appearance gallery: EMA weighted by detection confidence
        if detection.feat is not None:
            feat = detection.feat / (np.linalg.norm(detection.feat) + 1e-6)
            if conf >= 0.6:
                feat_weight = 1.0
            elif conf >= 0.3 and conf < 0.6:
                feat_weight = 0.8
            else:
                feat_weight = 0.4
            if self.features:
                adaptive_alpha = self.ema_alpha * feat_weight
                smooth_feat = adaptive_alpha * feat + (1 - adaptive_alpha) * self.features[-1]
                smooth_feat /= np.linalg.norm(smooth_feat) + 1e-6
                self.features.append(smooth_feat)
                if len(self.features) > 10:
                    self.features.pop(0)
            else:
                self.features = [feat]

        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2

    def apply_transfer_learning(
        self,
        frame_id: int | None = None,
        img_width: int | None = None,
        img_height: int | None = None,
    ) -> None:
        """Unmatched frame: let Primary learn from the Source prediction.

        Transfer learning runs only while the last high-confidence update is at most
        ``transfer_stale_gap`` frames old; otherwise the Primary velocity is set to zero
        (the track then stays in place until it is re-acquired or deleted). It is also
        skipped when the Source prediction is invalid (NaN, degenerate area/aspect ratio,
        centre far outside the image). ``transfer_active`` records whether it ran, and is
        one of the conditions for emitting a virtual box.
        """
        self.transfer_active = False

        if frame_id is not None:
            gap_since_hq = frame_id - self.last_high_quality_frame
            if gap_since_hq > self.transfer_stale_gap:
                self.primary_kf.x[4:8] *= 0.0
                self.time_since_update += 1
                return

        eta_pred = self.source_kf.x.copy()
        P_eta = self.source_kf.P.copy()
        if np.any(np.isnan(eta_pred)) or np.any(np.isinf(eta_pred)):
            self.time_since_update += 1
            return
        if np.any(np.isnan(P_eta)) or np.any(np.isinf(P_eta)):
            self.time_since_update += 1
            return

        # Validate the Source box before transferring it
        scale, ratio = eta_pred[2], eta_pred[3]
        if scale <= 0 or ratio <= 0 or scale > 4_000_000 or ratio > 100 or ratio < 0.01:
            self.time_since_update += 1
            return
        x, y, s, r = eta_pred[:4]
        w = np.sqrt(max(s * r, 0.0))
        h = np.sqrt(max(s / r, 0.0)) if r > 0 else 0.0
        box_width = abs((x + w / 2) - (x - w / 2))
        box_height = abs((y + h / 2) - (y - h / 2))
        if box_width * box_height < 100:
            self.time_since_update += 1
            return
        if box_width < 1.0 or box_height < 1.0:
            self.time_since_update += 1
            return
        aspect_check = box_width / box_height if box_height > 0 else 0
        if aspect_check < 0.1 or aspect_check > 10.0:
            self.time_since_update += 1
            return
        if img_width is not None and img_height is not None:
            if x < -w or x > img_width + w or y < -h or y > img_height + h:
                self.time_since_update += 1
                return

        # Primary learns from Source (Source state used as a virtual measurement)
        self.primary_kf.update(measurement=None, confidence=None, eta_pred=eta_pred, P_eta=P_eta)
        self.transfer_active = True
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()
        self.time_since_update += 1

        # Keep the appearance gallery alive during the gap (re-uses the last feature)
        if self.features:
            decay_factor = 0.95 ** self.time_since_update
            last_feat = self.features[-1].copy()
            virtual_feat = last_feat * decay_factor
            virtual_feat /= (np.linalg.norm(virtual_feat) + 1e-6)
            virtual_alpha = 0.3
            smooth_feat = virtual_alpha * virtual_feat + (1 - virtual_alpha) * self.features[-1]
            smooth_feat /= (np.linalg.norm(smooth_feat) + 1e-6)
            self.features.append(smooth_feat)
            if len(self.features) > 10:
                self.features.pop(0)

        if frame_id is not None:
            self.virtual_boxes.append((frame_id, self.primary_kf.x[:4].copy()))

    def camera_update(self, warp_matrix) -> None:
        """Apply the camera-motion (ECC) warp to the box of both filters."""
        M = _build_3x3_warp(warp_matrix)
        x1, y1, x2, y2 = self.to_tlbr()
        x1f = _ensure_scalar(x1)
        y1f = _ensure_scalar(y1)
        x2f = _ensure_scalar(x2)
        y2f = _ensure_scalar(y2)

        p1 = np.array([x1f, y1f, 1.0], dtype=np.float32).reshape(3, 1)
        p2 = np.array([x2f, y2f, 1.0], dtype=np.float32).reshape(3, 1)
        p1_t = M @ p1
        p2_t = M @ p2
        x1_, y1_ = float(p1_t[0, 0]), float(p1_t[1, 0])
        x2_, y2_ = float(p2_t[0, 0]), float(p2_t[1, 0])

        w = abs(x2_ - x1_)
        h = abs(y2_ - y1_)
        cx = (x1_ + x2_) / 2.0
        cy = (y1_ + y2_) / 2.0
        s = w * h
        r = (w / h) if (h > 1e-6) else 1.0
        new_state = np.array([cx, cy, s, r], dtype=np.float32)

        if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
            return
        if h < 1.0 or h > 10000.0 or w < 1.0 or w > 10000.0:
            return

        self.source_kf.x[:4] = new_state
        self.primary_kf.x[:4] = new_state
        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

    def is_confirmed(self) -> bool:
        """Whether the track has collected ``n_init`` matches."""
        return self.state == 2

    def is_deleted(self) -> bool:
        """Whether the miss counter exceeded ``max_age``."""
        return self.time_since_update > self._max_age
