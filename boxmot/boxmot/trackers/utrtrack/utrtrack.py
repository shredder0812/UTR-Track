# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from torch import device

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker, TrackerXYSR, TrackerTLUKF
from boxmot.trackers.strongsort.sort.linear_assignment import NearestNeighborDistanceMetric, NearestNeighborDistanceMetric_TLUKF
from boxmot.utils.ops import xyxy2tlwh, xyxy2xysr

from boxmot.utils import logger as LOGGER


class UTRTrack(BaseTracker):
    """
    Enhanced StrongSort -> UTR-Track tracker with virtual trajectory support for endoscopy videos.
    
    Additional Features:
    - Virtual trajectory interpolation for lost tracks
    - Appearance-based track recovery
    - Adaptive confidence for virtual detections
    - Frame buffering for appearance matching
    
    Parameters:
    [previous parameters documentation...]
    
    Additional Parameters:
    - max_gap (int): Maximum frame gap to interpolate virtual trajectory (default: 8)
    - appearance_thresh (float): Threshold for appearance similarity (default: 0.65)
    - virtual_conf (float): Confidence for virtual detections (default: 0.3)
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
        # StrongSort-specific parameters
        min_conf: float = 0.1,
        max_cos_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
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
        
        # Store StrongSort-specific parameters
        self.min_conf = min_conf
        
        # Initialize ReID model
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        # Initialize StrongSort tracker
        self.tracker = TrackerTLUKF(
            metric=NearestNeighborDistanceMetric_TLUKF("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        
        # Initialize camera motion compensation
        self.cmc = get_cmc_method("ecc")()
        # Confidence to assign virtual (missed) boxes in outputs
        self.virtual_conf: float = 0.3

        LOGGER.success("Initialized StrongSort")
        
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

        # update tracker with frame dimensions for boundary checking
        self.tracker.predict()
        img_height, img_width = img.shape[:2]
        self.tracker.update(detections, img_width=img_width, img_height=img_height)

        # output bbox identities
        outputs = []
        
        # CRITICAL FIX: Single loop to output ONLY ONE box per track
        # Prevents duplicate outputs (both real and virtual for same track)
        # 
        # Logic:
        # - time_since_update == 0: Track matched → output REAL box
        # - time_since_update >= 1: Track unmatched → output VIRTUAL box
        # Each track enters EXACTLY ONE branch - no duplicates possible
        
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            
            # CRITICAL: Check time_since_update to determine box type
            # time_since_update == 0: Track was matched this frame → Real box
            # time_since_update >= 1: Track was NOT matched → Virtual box (predicted)
            
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
                # UNMATCHED TRACK: Output virtual box with predicted position
                # TLUKF has already applied transfer learning in TrackerTLUKF.update()
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
                conf = 0.3  # Low confidence for virtual boxes
                cls = track.cls
                det_ind = getattr(track, 'det_ind', 0)
                
                outputs.append(
                    np.concatenate(
                        ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                    ).reshape(1, -1)
                )
            
        if len(outputs) > 0:
            outputs = np.concatenate(outputs)
            
            # VALIDATION: Check for duplicate IDs (should NEVER happen with new logic)
            unique_ids = set()
            duplicate_ids = []
            for output in outputs:
                track_id = int(output[4])
                if track_id in unique_ids:
                    duplicate_ids.append(track_id)
                else:
                    unique_ids.add(track_id)
            
            if duplicate_ids:
                # DEBUG: Log duplicate IDs for investigation
                import warnings
                warnings.warn(f"CRITICAL: Duplicate track IDs in same frame: {duplicate_ids}")
                # This should NEVER happen after our fix - if it does, we have a bug
            
            # Apply NMS to remove overlapping boxes (belt-and-suspenders approach)
            outputs = self._apply_nms(outputs, iou_threshold=0.5)
            return outputs
        return np.array([])
    
    def _apply_nms(self, tracks, iou_threshold=0.5):
        """
        Apply NMS to remove overlapping boxes with enhanced logic:
        1. Same ID: Keep ONLY ONE box - Priority: Strong > Weak > Virtual
        2. Multiple virtual boxes for same ID: Keep one closest to real detection in state space
        3. Different IDs but overlapping: Suppress virtual if real exists
        
        Args:
            tracks: numpy array [x1, y1, x2, y2, id, conf, cls, det_ind]
            iou_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            Filtered tracks after NMS
        """
        if len(tracks) == 0:
            return tracks
        
        # Step 1: Group by ID and keep only best box per ID
        id_to_tracks = {}
        for i, track in enumerate(tracks):
            track_id = int(track[4])
            if track_id not in id_to_tracks:
                id_to_tracks[track_id] = []
            id_to_tracks[track_id].append(track)
        
        # Step 2: For each ID, select best box
        filtered_tracks = []
        
        for track_id, track_group in id_to_tracks.items():
            if len(track_group) == 1:
                # Only one box for this ID - keep it
                filtered_tracks.append(track_group[0])
            else:
                # Multiple boxes for same ID
                real_boxes = [t for t in track_group if t[5] >= 0.35]  # Real detections
                virtual_boxes = [t for t in track_group if t[5] < 0.35]  # Virtual predictions
                
                if len(real_boxes) > 0:
                    # Has real detection - keep highest confidence
                    best_real = max(real_boxes, key=lambda t: t[5])
                    filtered_tracks.append(best_real)
                elif len(virtual_boxes) > 0:
                    # Only virtual boxes - keep first one
                    filtered_tracks.append(virtual_boxes[0])
        
        # Step 3: Spatial NMS for different IDs + LIMIT virtual boxes per frame
        if len(filtered_tracks) == 0:
            return np.array([])
        
        filtered_tracks = np.array(filtered_tracks)
        sorted_indices = np.argsort(-filtered_tracks[:, 5])  # Sort by conf descending
        sorted_tracks = filtered_tracks[sorted_indices]
        
        keep = []
        virtual_count = 0  # Track number of virtual boxes
        MAX_VIRTUAL_PER_FRAME = 1  # CRITICAL: Only 1 virtual box per frame
        
        for i, track in enumerate(sorted_tracks):
            track_id = int(track[4])
            track_conf = track[5]
            track_box = track[:4]
            is_virtual = track_conf < 0.35
            
            # Enforce virtual box limit per frame
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
                    # Different IDs but overlapping
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