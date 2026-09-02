from __future__ import absolute_import
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import numpy as np

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.utrtrack.sort import iou_matching, linear_assignment
from boxmot.trackers.utrtrack.sort.track import TrackTLUKF


class TrackerTLUKF:
    """
    Multi-target tracker using TL-UKF (Transfer Learning Unscented Kalman Filter).
    """
    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
        single_object_mode=False,
        reacquire_min_iou=0.05,
        reacquire_max_cost=0.55,
        reacquire_max_age=20,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.tracks = []
        self._next_id = 1
        self.cmc = get_cmc_method("ecc")()
        self.single_object_mode = single_object_mode
        self.reacquire_min_iou = reacquire_min_iou
        self.reacquire_max_cost = reacquire_max_cost
        self.reacquire_max_age = reacquire_max_age

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict()

    def increment_ages(self):
        """Increment ages and mark tracks as missed."""
        for track in self.tracks:
            track.age += 1
            if track.time_since_update > 0:
                track.time_since_update += 1

    def update(self, detections, frame_id=None, img_width=None, img_height=None):
        """
        Perform measurement update and track management with TL-UKF.

        Key mechanism:
        - Matched tracks: Updated with real detections (both Source and Primary)
        - Unmatched tracks: Apply transfer learning (Primary learns from Source)

        Args:
            detections: List of Detection objects
            frame_id: Current frame ID
            img_width: Frame width for boundary checking
            img_height: Frame height for boundary checking
        """
        # Run matching cascade (appearance + IOU)
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update matched tracks with real detections
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], frame_id=frame_id)

        # For unmatched tracks, apply transfer learning with boundary checking
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # Apply transfer learning: Primary learns from Source
            # Pass frame dimensions to prevent virtual boxes from running out of frame
            track.apply_transfer_learning(
                frame_id=frame_id,
                img_width=img_width,
                img_height=img_height
            )

        # For single-object videos, prioritize re-association to existing ID
        # before creating any new track id.
        if self.single_object_mode and unmatched_tracks and unmatched_detections:
            recovered_matches, unmatched_tracks, unmatched_detections = self._recover_single_object_matches(
                detections,
                unmatched_tracks,
                unmatched_detections,
            )
            for track_idx, detection_idx in recovered_matches:
                self.tracks[track_idx].update(detections[detection_idx], frame_id=frame_id)

        # Initialize new tracks from unmatched detections
        for detection_idx in unmatched_detections:
            if self.single_object_mode and self._has_live_confirmed_track():
                # Keep identity continuity: if one confirmed track is still alive,
                # do not spawn a competing ID from a single-frame mismatch.
                continue
            self._initiate_track(detections[detection_idx])

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        if self.single_object_mode and len(self.tracks) > 1:
            self._prune_to_single_track()

        # Update distance metric with ALL appearance features
        # Including features from virtual boxes to maintain similarity measurement
        # This prevents ID switches when object reappears after being missed
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # Collect ALL features from track (including virtual box features)
            features += track.features
            targets += [track.id for _ in track.features]

        # Update metric with latest features (both real and virtual)
        if len(features) > 0:
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets
            )

    def _has_live_confirmed_track(self):
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update <= self.max_age:
                return True
        return False

    @staticmethod
    def _det_tlbr(detection):
        x, y, w, h = detection.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _appearance_cost(self, track, detection):
        if not hasattr(detection, "feat") or detection.feat is None:
            return None
        if track.id not in self.metric.samples or len(self.metric.samples[track.id]) == 0:
            return None
        det_feat = np.asarray(detection.feat).reshape(1, -1)
        costs = self.metric.distance(det_feat, np.array([track.id]))
        return float(costs[0, 0])

    def _recover_single_object_matches(self, detections, unmatched_tracks, unmatched_detections):
        candidate_tracks = [
            t_idx
            for t_idx in unmatched_tracks
            if self.tracks[t_idx].is_confirmed() and self.tracks[t_idx].time_since_update <= self.reacquire_max_age
        ]

        if not candidate_tracks or not unmatched_detections:
            return [], unmatched_tracks, unmatched_detections

        recovered = []
        used_tracks = set()
        used_dets = set()

        # With one object, choose the best reassociation pair globally.
        best = None
        best_score = float("inf")
        for t_idx in candidate_tracks:
            track = self.tracks[t_idx]
            t_box = track.to_tlbr()
            for d_idx in unmatched_detections:
                det = detections[d_idx]
                d_box = self._det_tlbr(det)
                iou = self._bbox_iou(t_box, d_box)
                app_cost = self._appearance_cost(track, det)

                passes_iou = iou >= self.reacquire_min_iou
                passes_app = app_cost is not None and app_cost <= self.reacquire_max_cost
                if not (passes_iou or passes_app):
                    continue

                # Lower score is better. Prefer lower appearance cost and higher IoU.
                effective_app = app_cost if app_cost is not None else self.reacquire_max_cost
                score = effective_app - 0.5 * iou + 0.03 * track.time_since_update
                if score < best_score:
                    best_score = score
                    best = (t_idx, d_idx)

        if best is not None:
            t_idx, d_idx = best
            recovered.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        rem_tracks = [t for t in unmatched_tracks if t not in used_tracks]
        rem_dets = [d for d in unmatched_detections if d not in used_dets]
        return recovered, rem_tracks, rem_dets

    def _prune_to_single_track(self):
        # Keep the most reliable track when multiple tracks coexist in one-object videos.
        best_idx = None
        best_score = float("-inf")
        for i, track in enumerate(self.tracks):
            recent_bonus = 1000.0 if track.time_since_update == 0 else 0.0
            score = recent_bonus + 2.0 * float(track.hits) - 15.0 * float(track.time_since_update) + float(getattr(track, "conf", 0.0))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return
        self.tracks = [self.tracks[best_idx]]

    def _match(self, detections):
        """
        Matching cascade: appearance features + IOU matching.
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])

            cost_matrix = self.metric.distance(features, targets)

            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features (matching cascade)
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """
        Initialize new track from detection.

        Checks for overlapping old tracks and merges appearance features
        to maintain identity consistency and reduce ID switches.
        """
        # Convert detection tlwh to tlbr format
        x, y, w, h = detection.tlwh
        new_bbox = [x, y, x + w, y + h]
        tracks_to_remove = []
        merged_features = []

        # Check for overlapping tracks REGARDLESS of time_since_update
        # This prevents creating a new track when the object is already being tracked
        for i, track in enumerate(self.tracks):
            # Check ALL tracks, not just stale ones - in endoscopy, an object may be
            # missed briefly but still be tracked.
            track_bbox = track.to_tlbr()

            # Calculate IoU
            x1 = max(new_bbox[0], track_bbox[0])
            y1 = max(new_bbox[1], track_bbox[1])
            x2 = min(new_bbox[2], track_bbox[2])
            y2 = min(new_bbox[3], track_bbox[3])

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                bbox1_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                bbox2_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                union = bbox1_area + bbox2_area - intersection
                iou = intersection / union if union > 0 else 0

                # If IoU > 0.3 -> likely same object -> merge instead of creating new track
                if iou > 0.3:
                    tracks_to_remove.append(i)
                    # Merge features from old track to maintain appearance memory
                    if track.features:
                        merged_features.extend(track.features[-5:])  # Keep last 5 features

        # Create new track
        new_track = TrackTLUKF(
            detection,
            self._next_id,
            self.n_init,
            self.max_age,
            self.ema_alpha,
        )

        # Merge features from overlapping old tracks
        # This maintains appearance memory and reduces ID switches
        if merged_features and hasattr(detection, 'feat') and detection.feat is not None:
            # Add merged features to new track's feature gallery
            # This helps similarity measurement recognize it as the same object
            for old_feat in merged_features:
                if len(new_track.features) < 10:
                    new_track.features.append(old_feat)

        # Remove overlapping old tracks AFTER merging features
        for i in sorted(tracks_to_remove, reverse=True):
            del self.tracks[i]

        self.tracks.append(new_track)
        self._next_id += 1
