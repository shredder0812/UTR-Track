"""
Track classes for EndoOcSort tracker using XYSR-TLUKF motion model.
"""

import numpy as np
from boxmot.motion.kalman_filters.aabb.xysr_tlukf import KalmanFilterXYSR_TLUKF, xyxy2xysr, xysr2xyxy


class TrackState:
    """Track state enumeration."""
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class TrackEndoOcSort:
    """
    Single target track với XYSR-TLUKF Kalman filter.
    
    State space: [x, y, s, r, vx, vy, vs, vr]
    - x, y: center position
    - s: scale (area)
    - r: aspect ratio
    - vx, vy, vs, vr: velocities
    
    Attributes:
        track_id: Unique track identifier
        hits: Number of measurement updates
        age: Total frames since first occurrence
        time_since_update: Frames since last update
        state: TrackState (Tentative, Confirmed, Deleted)
        conf: Confidence score
        cls: Class ID
        det_ind: Detection index
    """
    
    def __init__(self, bbox_xyxy, cls, det_ind, track_id, n_init=3, max_age=30, 
                 is_source=False, ema_alpha=0.9):
        """
        Initialize track from detection.
        
        Args:
            bbox_xyxy: Bounding box [x1, y1, x2, y2, conf]
            cls: Class ID
            det_ind: Detection index
            track_id: Unique track ID
            n_init: Number of hits before confirmed
            max_age: Maximum age before deletion
            is_source: True if Source tracker, False if Primary
            ema_alpha: EMA smoothing factor for features
        """
        # Track identity
        self.track_id = track_id
        self.cls = cls
        self.det_ind = det_ind
        self.conf = bbox_xyxy[-1] if len(bbox_xyxy) > 4 else 1.0
        
        # Track state
        self.state = TrackState.Tentative
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        
        # Kalman filter
        self.kf = KalmanFilterXYSR_TLUKF(is_source=is_source)
        bbox_xysr = xyxy2xysr(bbox_xyxy[:4])
        self.mean, self.covariance = self.kf.initiate(bbox_xysr)
        
        # Feature tracking
        self.features = []
        self.ema_alpha = ema_alpha
        self.smooth_feat = None
        
        # Velocity tracking (for OcSort-style association)
        self.velocity = None
        self.last_observation = bbox_xyxy[:5] if len(bbox_xyxy) >= 5 else np.r_[bbox_xyxy, [1.0]]
        self.observations = {0: self.last_observation}
        self.history_observations = [self.last_observation]
        
    def to_xyxy(self):
        """Get current position in XYXY format."""
        xysr = self.mean[:4].flatten()
        return xysr2xyxy(xysr)
    
    def to_xysr(self):
        """Get current position in XYSR format."""
        return self.mean[:4].flatten()
    
    def predict(self):
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox_xyxy=None, cls=None, det_ind=None, 
               eta_pred=None, P_eta=None):
        """
        Update track with measurement and/or transfer learning.
        
        Args:
            bbox_xyxy: Detection [x1, y1, x2, y2, conf] or None
            cls: Class ID
            det_ind: Detection index
            eta_pred: Source tracker prediction (for Primary)
            P_eta: Source tracker covariance (for Primary)
        """
        # Update class and detection info
        if cls is not None:
            self.cls = cls
        if det_ind is not None:
            self.det_ind = det_ind
            
        # Update with measurement
        if bbox_xyxy is not None:
            self.conf = bbox_xyxy[-1] if len(bbox_xyxy) > 4 else 1.0
            bbox_xysr = xyxy2xysr(bbox_xyxy[:4])
            
            # Kalman update
            self.mean, self.covariance = self.kf.update(
                measurement=bbox_xysr,
                confidence=self.conf,
                eta_pred=eta_pred,
                P_eta=P_eta
            )
            
            # Update observations
            self.last_observation = bbox_xyxy[:5] if len(bbox_xyxy) >= 5 else np.r_[bbox_xyxy, [1.0]]
            self.observations[self.age] = self.last_observation
            self.history_observations.append(self.last_observation)
            
            # Update velocity (for OcSort compatibility)
            if len(self.history_observations) >= 2:
                prev_obs = self.history_observations[-2]
                curr_obs = self.last_observation
                cx1, cy1 = (prev_obs[0] + prev_obs[2]) / 2, (prev_obs[1] + prev_obs[3]) / 2
                cx2, cy2 = (curr_obs[0] + curr_obs[2]) / 2, (curr_obs[1] + curr_obs[3]) / 2
                speed = np.array([cy2 - cy1, cx2 - cx1])
                norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
                self.velocity = speed / norm
            
            self.hits += 1
            self.time_since_update = 0
        else:
            # No measurement - only TL update
            if eta_pred is not None:
                self.mean, self.covariance = self.kf.update(
                    measurement=None,
                    confidence=None,
                    eta_pred=eta_pred,
                    P_eta=P_eta
                )
        
        # Update state
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        """Returns True if this track is deleted."""
        return self.state == TrackState.Deleted
    
    def update_features(self, feat):
        """
        Update feature vector with EMA smoothing.
        
        Args:
            feat: New feature vector
        """
        if feat is None:
            return
            
        feat = np.asarray(feat).flatten()
        self.features.append(feat)
        
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.ema_alpha * self.smooth_feat + (1 - self.ema_alpha) * feat
    
    def get_feature(self):
        """Get current smoothed feature."""
        return self.smooth_feat
    
    def gating_distance(self, measurements, only_position=False, metric='maha'):
        """
        Compute gating distance between track and measurements.
        
        Args:
            measurements: Array of measurements in XYSR format
            only_position: If True, only use position
            metric: Distance metric
            
        Returns:
            Array of distances
        """
        return self.kf.gating_distance(
            self.mean,
            self.covariance,
            measurements,
            only_position=only_position,
            metric=metric
        )
