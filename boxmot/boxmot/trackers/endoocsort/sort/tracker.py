"""
Tracker implementation for EndoOcSort using XYSR-TLUKF with Transfer Learning.
Kết hợp OcSort logic với TLUKF motion model.
"""

from __future__ import absolute_import
import numpy as np

from boxmot.trackers.endoocsort.sort import linear_assignment, iou_matching
from boxmot.trackers.endoocsort.sort.detection import Detection
from boxmot.trackers.endoocsort.sort.track import TrackEndoOcSort, TrackState
from boxmot.motion.kalman_filters.aabb.xysr_tlukf import xyxy2xysr


class TrackerEndoOcSort:
    """
    Multi-target tracker using XYSR-TLUKF with OcSort-style association.
    
    TLUKF Key Features:
    - Dual tracker system: Source (high-conf) + Primary (all detections)
    - Transfer Learning: Primary learns from Source during gaps
    - Virtual boxes: Maintain tracking during occlusions
    
    OcSort Features:
    - Velocity-based association
    - Observation history tracking
    - BYTE-style second-stage matching
    """
    
    def __init__(self, metric=None, max_iou_dist=0.7, max_age=30, n_init=3,
                 ema_alpha=0.9, mc_lambda=0.995, delta_t=3, inertia=0.2,
                 use_byte=True, min_conf=0.1):
        """
        Args:
            metric: Appearance metric (ReID model)
            max_iou_dist: Max IOU distance for association
            max_age: Maximum frames to keep track alive
            n_init: Number of hits before track is confirmed
            ema_alpha: EMA smoothing for appearance features
            mc_lambda: Momentum for metric learning
            delta_t: Observation history window
            inertia: Weight for velocity in association
            use_byte: Use BYTE association for low-conf detections
            min_conf: Minimum confidence for second-stage association
        """
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.min_conf = min_conf
        
        # Track management
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0
    
    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict()
    
    def increment_ages(self):
        """Increment ages and mark tracks as missed."""
        for track in self.tracks:
            if track.time_since_update > 0:
                track.time_since_update += 1
                track.mark_missed()
    
    def update(self, detections_high, detections_low=None, frame_id=None, 
               img=None, img_width=None, img_height=None):
        """
        Update tracker with detections.
        
        TLUKF Strategy:
        - Source tracker: Uses only high-conf detections (conf >= 0.6)
        - Primary tracker: Uses all detections (conf >= 0.3)
        - Unmatched Primary tracks learn from Source predictions
        
        Args:
            detections_high: High-confidence detections for Source tracker
            detections_low: Low-confidence detections for second-stage matching
            frame_id: Current frame ID
            img: Current frame image
            img_width: Frame width
            img_height: Frame height
        """
        self.frame_count += 1
        
        # Convert to Detection objects
        dets_high = self._create_detections(detections_high)
        dets_low = self._create_detections(detections_low) if detections_low is not None else []
        
        # First matching cascade: High-confidence detections
        matches, unmatched_tracks, unmatched_dets_high = self._match_cascade(
            dets_high, img
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                bbox_xyxy=np.r_[dets_high[det_idx].xyxy, dets_high[det_idx].confidence],
                cls=dets_high[det_idx].cls,
                det_ind=dets_high[det_idx].det_ind
            )
        
        # Second-stage matching: BYTE-style low-conf association
        if self.use_byte and len(dets_low) > 0 and len(unmatched_tracks) > 0:
            matches_byte, unmatched_tracks, unmatched_dets_low = self._match_byte(
                unmatched_tracks, dets_low
            )
            
            for track_idx, det_idx in matches_byte:
                self.tracks[track_idx].update(
                    bbox_xyxy=np.r_[dets_low[det_idx].xyxy, dets_low[det_idx].confidence],
                    cls=dets_low[det_idx].cls,
                    det_ind=dets_low[det_idx].det_ind
                )
        
        # Third-stage: Re-association with observation history (OcSort)
        if len(unmatched_tracks) > 0 and len(unmatched_dets_high) > 0:
            matches_obs, unmatched_tracks, unmatched_dets_high = self._match_observation(
                unmatched_tracks, unmatched_dets_high
            )
            
            for track_idx, det_idx in matches_obs:
                self.tracks[track_idx].update(
                    bbox_xyxy=np.r_[dets_high[det_idx].xyxy, dets_high[det_idx].confidence],
                    cls=dets_high[det_idx].cls,
                    det_ind=dets_high[det_idx].det_ind
                )
        
        # TLUKF: Apply Transfer Learning for unmatched tracks
        # Primary tracks learn from Source predictions
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            
            # Find corresponding Source tracker prediction if exists
            # (In practice, this would come from a parallel Source tracker)
            # For now, just mark as missed
            track.update(bbox_xyxy=None)
        
        # Initialize new tracks from unmatched detections
        for det_idx in unmatched_dets_high:
            self._initiate_track(dets_high[det_idx])
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Update appearance metric
        if self.metric is not None:
            self._update_metric()
    
    def _create_detections(self, dets):
        """Convert detection array to Detection objects."""
        if dets is None or len(dets) == 0:
            return []
        
        detection_list = []
        for i, det in enumerate(dets):
            if len(det) < 6:
                continue
            xyxy = det[:4]
            conf = det[4]
            cls = int(det[5])
            det_ind = int(det[6]) if len(det) > 6 else i
            
            detection_list.append(Detection(xyxy, conf, cls, det_ind=det_ind))
        
        return detection_list
    
    def _match_cascade(self, detections, img):
        """
        Main matching using appearance + IOU cascade.
        """
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []
        
        # Split confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Appearance-based matching for confirmed tracks
        if self.metric is not None and len(confirmed_tracks) > 0:
            matches_a, unmatched_tracks_a, unmatched_dets = self._match_appearance(
                confirmed_tracks, detections
            )
        else:
            matches_a = []
            unmatched_tracks_a = confirmed_tracks
            unmatched_dets = list(range(len(detections)))
        
        # IOU-based matching for unconfirmed + recent unmatched tracks
        iou_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        
        matches_b, unmatched_tracks_b, unmatched_dets = self._match_iou(
            iou_candidates, detections, unmatched_dets
        )
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _match_appearance(self, track_indices, detections):
        """Appearance-based matching using cascade."""
        # This would use ReID features - simplified for now
        return self._match_iou(track_indices, detections)
    
    def _match_iou(self, track_indices, detections, det_indices=None):
        """IOU-based matching."""
        if det_indices is None:
            det_indices = list(range(len(detections)))
        
        if len(track_indices) == 0 or len(det_indices) == 0:
            return [], track_indices, det_indices
        
        # Compute IOU cost matrix
        cost_matrix = np.zeros((len(track_indices), len(det_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track_bbox = self.tracks[track_idx].to_xyxy()
            
            for j, det_idx in enumerate(det_indices):
                det_bbox = detections[det_idx].xyxy
                
                # Compute IOU
                iou = self._compute_iou(track_bbox, det_bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        # Solve assignment
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_dets = list(det_indices)
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < (1.0 - self.max_iou_dist):
                track_idx = track_indices[row]
                det_idx = det_indices[col]
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _match_byte(self, track_indices, detections):
        """BYTE-style second-stage matching with low-conf detections."""
        return self._match_iou(track_indices, detections)
    
    def _match_observation(self, track_indices, detections):
        """OcSort-style matching using observation history."""
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        
        # Use last observations for matching
        cost_matrix = np.zeros((len(track_indices), len(detections)))
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            if track.last_observation is None:
                cost_matrix[i, :] = 1.0
                continue
            
            last_obs = track.last_observation[:4]
            
            for j, det in enumerate(detections):
                iou = self._compute_iou(last_obs, det.xyxy)
                cost_matrix[i, j] = 1.0 - iou
        
        # Solve assignment
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_dets = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < (1.0 - self.max_iou_dist):
                track_idx = track_indices[row]
                matches.append((track_idx, col))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(col)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IOU between two bboxes in XYXY format."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _initiate_track(self, detection):
        """Initialize new track from detection."""
        bbox_xyxy = np.r_[detection.xyxy, detection.confidence]
        
        track = TrackEndoOcSort(
            bbox_xyxy=bbox_xyxy,
            cls=detection.cls,
            det_ind=detection.det_ind,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            is_source=False,  # All tracks are Primary for now
            ema_alpha=self.ema_alpha
        )
        
        self.tracks.append(track)
        self._next_id += 1
    
    def _update_metric(self):
        """Update appearance metric with current track features."""
        if self.metric is None:
            return
        
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            feat = track.get_feature()
            if feat is not None:
                features.append(feat)
                targets.append(track.track_id)
        
        if len(features) > 0:
            self.metric.partial_fit(
                np.asarray(features),
                np.asarray(targets),
                active_targets
            )
