from __future__ import absolute_import
# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import numpy as np

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort import iou_matching, linear_assignment
from boxmot.trackers.strongsort.sort.track import Track, TrackXYSR, TrackTLUKF
from boxmot.utils.matching import chi2inv95


class TrackerTLUKF:
    """
    Multi-target tracker sử dụng TLUKF (Unscented Kalman Filter).
    """
    def __init__(self, metric, max_iou_dist=0.9, max_age=30, n_init=3, _lambda=0, ema_alpha=0.9, mc_lambda=0.995):
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
        Perform measurement update and track management with TLUKF.
        
        TLUKF Key Innovation:
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
        
        # TLUKF: For unmatched tracks, apply transfer learning with boundary checking
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # Apply transfer learning: Primary learns from Source
            # Pass frame dimensions to prevent virtual boxes from running out of frame
            track.apply_transfer_learning(
                frame_id=frame_id,
                img_width=img_width,
                img_height=img_height
            )
        
        # Initialize new tracks from unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # CRITICAL FIX: Update distance metric with ALL appearance features
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

    def _match(self, detections):
        """
        Matching cascade: appearance features + IOU matching.
        Same logic as TrackerXYSR but adapted for TLUKF tracks.
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            
            # DEBUG: Check if features are normalized
            # for i, feat in enumerate(features):
            #     norm = np.linalg.norm(feat)
            #     if abs(norm - 1.0) > 0.1:
            #         print(f"WARNING: Detection {i} feature not normalized! norm={norm:.3f}")
            
            cost_matrix = self.metric.distance(features, targets)
            
            # DEBUG: Log matching costs
            # if cost_matrix.size > 0:
            #     min_cost = cost_matrix.min()
            #     max_cost = cost_matrix.max()
            #     print(f"Cost matrix: min={min_cost:.3f}, max={max_cost:.3f}, threshold={self.metric.matching_threshold}")
            
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
        
        Enhanced: Check for overlapping old tracks and merge appearance features
        to maintain identity consistency and reduce ID switches.
        """
        # Convert detection tlwh to tlbr format
        x, y, w, h = detection.tlwh
        new_bbox = [x, y, x + w, y + h]
        tracks_to_remove = []
        merged_features = []
        
        # CRITICAL: Check for overlapping tracks REGARDLESS of time_since_update
        # This prevents creating new track when object is already being tracked
        for i, track in enumerate(self.tracks):
            # Check ALL tracks, not just stale ones
            # Reason: In endoscopy, object may be missed briefly but still being tracked
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
                
                # CRITICAL: Lower IoU threshold for detection
                # If IoU > 0.3 → likely same object → merge instead of creating new track
                if iou > 0.3:  # Lowered from 0.5 to 0.3
                    tracks_to_remove.append(i)
                    # Merge features from old track to maintain appearance memory
                    if track.features:
                        merged_features.extend(track.features[-5:])  # Keep last 5 features (increased from 3)
        
        # Create new track
        new_track = TrackTLUKF(
            detection,
            self._next_id,
            self.n_init,
            self.max_age,
            self.ema_alpha,
        )
        
        # CRITICAL FIX: Merge features from overlapping old tracks
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


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
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

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
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

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
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
        self.tracks.append(
            Track(
                detection,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
            )
        )
        self._next_id += 1

class TrackerXYSR:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
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

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
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

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
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
        self.tracks.append(
            TrackXYSR(
                detection,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
            )
        )
        self._next_id += 1
