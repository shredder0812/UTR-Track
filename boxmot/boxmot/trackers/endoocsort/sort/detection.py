# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Detection class for EndoOcSort tracker.
"""

import numpy as np
from boxmot.motion.kalman_filters.aabb.xysr_tlukf import xyxy2xysr


class Detection:
    """
    Single object detection.
    
    Attributes:
        xyxy: Bounding box [x1, y1, x2, y2]
        xysr: Bounding box [x, y, s, r] (center, scale, ratio)
        confidence: Detection confidence
        cls: Class ID
        feature: Appearance feature vector
        det_ind: Detection index
    """
    
    def __init__(self, xyxy, confidence, cls, feature=None, det_ind=None):
        """
        Args:
            xyxy: Bounding box [x1, y1, x2, y2]
            confidence: Confidence score
            cls: Class ID
            feature: Optional appearance feature
            det_ind: Detection index
        """
        self.xyxy = np.asarray(xyxy).flatten()[:4]
        self.confidence = float(confidence)
        self.cls = int(cls)
        self.feature = feature
        self.det_ind = det_ind if det_ind is not None else -1
        
        # Convert to XYSR format
        self.xysr = xyxy2xysr(self.xyxy)
    
    def to_xyxy(self):
        """Get bbox in XYXY format."""
        return self.xyxy.copy()
    
    def to_xysr(self):
        """Get bbox in XYSR format."""
        return self.xysr.copy()
    
    def to_tlwh(self):
        """Get bbox in TLWH format [x1, y1, width, height]."""
        ret = self.xyxy.copy()
        ret[2:] = ret[2:] - ret[:2]
        return ret
    
    def to_xyah(self):
        """Get bbox in XYAH format [x_center, y_center, aspect_ratio, height]."""
        ret = self.to_tlwh().copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def __repr__(self):
        return f"Detection(xyxy={self.xyxy}, conf={self.confidence:.2f}, cls={self.cls})"
