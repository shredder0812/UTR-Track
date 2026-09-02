# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import numpy as np

from boxmot.motion.kalman_filters.aabb.tlukf import TLUKFTracker


def _ensure_scalar(v):
    """Return float scalar from v (which may be numpy scalar, Python scalar, or 0-d array)."""
    try:
        return float(np.asarray(v).item())
    except Exception:
        # fallback: try float conversion
        return float(v)


def _build_3x3_warp(warp):
    """
    Accept either:
      - warp = (a, b) where a,b are row-like length-3,
      - warp = 3x3 array-like
    Return np.ndarray shape (3,3) dtype float32.
    """
    warp = np.asarray(warp)
    if warp.shape == (2, 3):  # two rows provided
        a = warp[0]
        b = warp[1]
        third = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        M = np.vstack([a, b, third]).astype(np.float32)
        return M
    elif warp.shape == (2,) and hasattr(warp[0], "__len__") and len(warp[0]) == 3:
        # sometimes provided as tuple/list: (a,b)
        a, b = warp[0], warp[1]
        third = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        M = np.vstack([a, b, third]).astype(np.float32)
        return M
    else:
        # try treat as full 3x3
        M = np.asarray(warp, dtype=np.float32)
        if M.shape == (3, 3):
            return M
        raise ValueError(f"Unsupported warp_matrix shape: {warp.shape}")


class TrackTLUKF:
    """
    UTRTrack track: Transfer Learning Unscented Kalman Filter (TL-UKF).
    Dual-tracker architecture: Source (teacher) and Primary (student).
    State: [x, y, a, h, vx, vy, va, vh]
    """
    def __init__(self, detection, id, n_init, max_age, ema_alpha, high_conf_threshold=0.6):
        self.id = id
        # Normalize input bbox: [x1, y1, x2, y2] or [cx, cy, a, h]
        if hasattr(detection, 'to_xyah'):
            bbox = detection.to_xyah()
        elif hasattr(detection, 'tlwh'):
            x1, y1, w, h = detection.tlwh
            cx = x1 + w / 2
            cy = y1 + h / 2
            a = w / h if h > 0 else 1.0
            bbox = np.array([cx, cy, a, h], dtype=np.float32)
        else:
            bbox = np.asarray(detection[:4], dtype=np.float32)
        self.bbox = bbox
        self.conf = getattr(detection, 'conf', 1.0)
        self.cls = getattr(detection, 'cls', 0)
        self.det_ind = getattr(detection, 'det_ind', 0)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.state = 1  # Tentative
        self.features = []

        # Class voting for stability (prevent temporary misclassification)
        self.class_history = [self.cls]  # Track last N classes
        self.class_vote_window = 200  # Vote over last 200 frames
        self.stable_cls = self.cls  # Class determined by majority voting
        if hasattr(detection, 'feat') and detection.feat is not None:
            feat = detection.feat / (np.linalg.norm(detection.feat) + 1e-6)
            self.features = [feat]
        self._n_init = n_init
        self._max_age = max_age
        self.high_conf_threshold = high_conf_threshold

        # Dual-tracker architecture
        self.source_kf = TLUKFTracker(is_source=True)   # Teacher: only high-quality updates
        self.primary_kf = TLUKFTracker(is_source=False)  # Student: all updates + transfer learning

        # Initialize both trackers
        self.source_kf.initiate(self.bbox)
        self.primary_kf.initiate(self.bbox)

        # Use primary tracker for main state
        self.kf = self.primary_kf  # For compatibility
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

        # Virtual trajectory tracking
        self.virtual_boxes = []
        self.last_high_quality_frame = 0
        self.last_real_frame = 0
        self.transfer_active = False

        # Static scene detection (for video pause handling)
        self.last_position = bbox[:2].copy()  # [x, y]
        self.static_frame_count = 0
        self.position_threshold = 1.0  # pixels - if movement < this, consider static

    def to_tlwh(self):
        x, y, a, h = self.kf.x[:4]
        w = a * h
        tl_x = x - w / 2
        tl_y = y - h / 2
        return np.array([tl_x, tl_y, w, h], dtype=np.float32)

    def to_tlbr(self):
        tlwh = self.to_tlwh()
        x1, y1, w, h = tlwh
        x2 = x1 + w
        y2 = y1 + h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self):
        """
        Predict next state with static scene detection.
        If scene is static (e.g., video pause), dampen velocity to prevent drift.
        """
        # Predict both Source and Primary trackers FIRST
        self.source_kf.predict()
        self.primary_kf.predict()

        # NOW check if position has changed significantly (AFTER prediction)
        current_pos = self.primary_kf.x[:2].copy()
        pos_change = np.linalg.norm(current_pos - self.last_position)

        if pos_change < self.position_threshold:
            # Likely static scene - dampen velocities
            self.static_frame_count += 1

            # After 3 static frames, heavily dampen velocities AND REVERT POSITION
            if self.static_frame_count >= 3:
                # Revert position to last known position (prevent drift)
                self.source_kf.x[:2] = self.last_position.copy()
                self.primary_kf.x[:2] = self.last_position.copy()
                # Zero out all velocities
                self.source_kf.x[4:8] = 0.0
                self.primary_kf.x[4:8] = 0.0
        else:
            # Movement detected - reset counter and update last position
            self.static_frame_count = 0
            self.last_position = current_pos.copy()

        # Update main state from primary
        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

        self.age += 1
        self.time_since_update += 1

    def update(self, detection, frame_id=None):
        """
        Update track with detection.
        Updates both Source and Primary trackers intelligently.

        Stores appearance features from ALL detections (strong, weak, virtual)
        to improve similarity measurement and reduce ID switches.
        """
        if hasattr(detection, 'to_xyah'):
            bbox = detection.to_xyah()
        elif hasattr(detection, 'tlwh'):
            x1, y1, w, h = detection.tlwh
            cx = x1 + w / 2
            cy = y1 + h / 2
            a = w / h if h > 0 else 1.0
            bbox = np.array([cx, cy, a, h], dtype=np.float32)
        else:
            bbox = np.asarray(detection[:4], dtype=np.float32)

        conf = getattr(detection, 'conf', 1.0)

        # Reset static frame counter on new detection AND update last position
        self.static_frame_count = 0
        # last_position is updated AFTER the KF update below (in state space)

        # Always update Primary with actual measurement
        self.primary_kf.update(measurement=bbox, confidence=conf)
        self.transfer_active = False
        if frame_id is not None:
            self.last_real_frame = frame_id

        # Update Source ONLY with high-quality detections
        if conf >= self.high_conf_threshold:
            self.source_kf.update(measurement=bbox, confidence=conf)
            if frame_id is not None:
                self.last_high_quality_frame = frame_id
            # Mark that we have recent high-quality data
            self.has_recent_hq = True

        # Update main state from primary
        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()
        self.bbox = bbox

        # Update last_position AFTER update (in state space)
        self.last_position = self.primary_kf.x[:2].copy()
        self.hits += 1
        self.time_since_update = 0

        # Store detection info for track consistency
        self.conf = conf

        # Class voting mechanism: Update class history and compute stable class
        if hasattr(detection, 'cls'):
            # Add new class to history
            self.class_history.append(detection.cls)

            # Keep only last N classes
            if len(self.class_history) > self.class_vote_window:
                self.class_history.pop(0)

            # Majority voting: Find most common class in history
            class_counts = {}
            for cls in self.class_history:
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Get class with highest count (majority vote)
            self.stable_cls = max(class_counts, key=class_counts.get)

            # Update cls to stable_cls (not the raw detection class)
            self.cls = self.stable_cls

        if hasattr(detection, 'det_ind'):
            self.det_ind = detection.det_ind

        # Update appearance features from ALL detections (not just high-conf)
        # This improves similarity measurement and reduces ID switches
        if hasattr(detection, 'feat') and detection.feat is not None:
            feat = detection.feat / (np.linalg.norm(detection.feat) + 1e-6)

            # Weight features based on confidence
            # High conf (>=0.6): Full weight
            # Medium conf (0.3-0.6): 80% weight
            # Low conf (<0.3): 40% weight
            if conf >= 0.6:
                feat_weight = 1.0
            elif conf >= 0.3 and conf < 0.6:
                feat_weight = 0.8
            else:
                feat_weight = 0.4

            if self.features:
                # Adaptive EMA based on confidence and feature weight
                # High confidence -> more trust in new feature (higher alpha)
                # Low confidence -> more trust in existing features (lower alpha)
                # Formula: smooth_feat = alpha * new_feat + (1 - alpha) * old_feat
                adaptive_alpha = self.ema_alpha * feat_weight
                smooth_feat = adaptive_alpha * feat + (1 - adaptive_alpha) * self.features[-1]
                smooth_feat /= np.linalg.norm(smooth_feat) + 1e-6

                # Keep multiple features in gallery (not just 1)
                self.features.append(smooth_feat)

                # Limit gallery size but keep more than 1
                if len(self.features) > 10:  # Keep last 10 features
                    self.features.pop(0)
            else:
                self.features = [feat]

        # State transition
        if self.state == 1 and self.hits >= self._n_init:
            self.state = 2  # Confirmed

    def apply_transfer_learning(self, frame_id=None, img_width=None, img_height=None):
        """
        Transfer learning from Source to Primary when no detection matched.
        This is the core innovation of TL-UKF - non-linear motion prediction.

        Only apply if Source tracker has recent high-quality updates.

        Includes boundary checks to prevent virtual boxes from running out of frame.
        """
        # Default to disabled; set True only when transfer update is applied successfully.
        self.transfer_active = False

        # Check if Source tracker has been updated recently with high-quality data
        if frame_id is not None and hasattr(self, 'last_high_quality_frame'):
            gap_since_hq = frame_id - self.last_high_quality_frame
            # Only use Source knowledge if it's fresh (within 5 frames)
            if gap_since_hq > 5:
                # Source is too stale, just use Primary's own prediction
                self.primary_kf.x[4:8] *= 0.0
                self.time_since_update += 1
                return

        # Get knowledge from Source tracker
        eta_pred = self.source_kf.x.copy()
        P_eta = self.source_kf.P.copy()

        # Validate Source tracker state before transfer
        if np.any(np.isnan(eta_pred)) or np.any(np.isinf(eta_pred)):
            # Source state is invalid, skip transfer learning
            self.time_since_update += 1
            return

        if np.any(np.isnan(P_eta)) or np.any(np.isinf(P_eta)):
            # Source covariance is invalid, skip transfer learning
            self.time_since_update += 1
            return

        # Validate box dimensions (aspect ratio and height should be reasonable)
        aspect_ratio = eta_pred[2]
        height = eta_pred[3]
        if aspect_ratio <= 0 or height <= 0 or height > 10000 or aspect_ratio > 100:
            # Invalid dimensions, skip transfer learning
            self.time_since_update += 1
            return

        # Check if predicted box is within frame bounds
        x, y, a, h = eta_pred[:4]
        w = a * h
        x1_pred = x - w / 2
        y1_pred = y - h / 2
        x2_pred = x + w / 2
        y2_pred = y + h / 2

        # Validate box dimensions before proceeding
        # Check if box area is too small (degenerate box)
        box_width = abs(x2_pred - x1_pred)
        box_height = abs(y2_pred - y1_pred)
        box_area = box_width * box_height
        min_area = 100  # Minimum box area in pixels
        if box_area < min_area:
            # Box too small, likely corrupted - skip transfer learning
            self.time_since_update += 1
            return

        # Check if box has degenerate coordinates (points too close)
        epsilon = 1.0  # Minimum 1 pixel difference
        if box_width < epsilon or box_height < epsilon:
            # Degenerate box (collapsed to line or point) - skip
            self.time_since_update += 1
            return

        # Check aspect ratio is reasonable
        aspect_check = box_width / box_height if box_height > 0 else 0
        if aspect_check < 0.1 or aspect_check > 10.0:
            # Unreasonable aspect ratio - skip
            self.time_since_update += 1
            return

        # If frame dimensions provided, check boundaries
        if img_width is not None and img_height is not None:
            # Check if box center is completely out of frame
            if x < -w or x > img_width + w or y < -h or y > img_height + h:
                # Box has moved completely out of frame - delete track
                self.time_since_update += 1
                return

            # Check if box is mostly out of frame (>70% outside)
            visible_x1 = max(0, x1_pred)
            visible_y1 = max(0, y1_pred)
            visible_x2 = min(img_width, x2_pred)
            visible_y2 = min(img_height, y2_pred)

            if visible_x2 > visible_x1 and visible_y2 > visible_y1:
                visible_area = (visible_x2 - visible_x1) * (visible_y2 - visible_y1)
                total_area = w * h
                visible_ratio = visible_area / total_area if total_area > 0 else 0

                if visible_ratio < 0.3:  # Less than 30% visible
                    # Box mostly out of frame - dampen velocity instead of deleting
                    eta_pred[4:8] *= 0.1  # Reduce velocity by 90%

            # Clamp velocity to reasonable bounds (prevent running away)
            max_velocity_x = img_width * 0.05  # Max 5% of frame width per frame
            max_velocity_y = img_height * 0.05  # Max 5% of frame height per frame
            eta_pred[4] = np.clip(eta_pred[4], -max_velocity_x, max_velocity_x)
            eta_pred[5] = np.clip(eta_pred[5], -max_velocity_y, max_velocity_y)

        # Check velocity magnitude - if too high, dampen it
        velocity_magnitude = np.sqrt(eta_pred[4]**2 + eta_pred[5]**2)
        max_reasonable_velocity = height * 0.5  # Max 50% of box height per frame
        if velocity_magnitude > max_reasonable_velocity:
            # Scale down velocity
            scale = max_reasonable_velocity / velocity_magnitude
            eta_pred[4] *= scale
            eta_pred[5] *= scale

        # Primary learns from Source (virtual measurement from teacher)
        # This provides non-linear motion tracking
        self.primary_kf.update(
            measurement=None,
            confidence=None,
            eta_pred=eta_pred,
            P_eta=P_eta
        )
        self.transfer_active = True

        # Update main state from Primary
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

        # Mark as unmatched but still tracked via transfer learning
        self.time_since_update += 1

        # Maintain feature gallery for virtual boxes to improve similarity measurement
        # This reduces ID switches by keeping appearance memory during missed detections
        if self.features:
            # Virtual box: propagate last feature with reduced confidence
            # Use exponential decay based on time_since_update
            decay_factor = 0.95 ** self.time_since_update  # Decay: 0.95, 0.90, 0.86, ...

            # Apply decay to last feature (simulating appearance persistence)
            last_feat = self.features[-1].copy()
            virtual_feat = last_feat * decay_factor

            # Normalize to maintain unit length
            virtual_feat /= (np.linalg.norm(virtual_feat) + 1e-6)

            # Update feature gallery with virtual feature
            # Use low EMA alpha to maintain stability (more weight on history)
            virtual_alpha = 0.3  # Low alpha = trust historical features more
            if len(self.features) > 0:
                smooth_feat = virtual_alpha * virtual_feat + (1 - virtual_alpha) * self.features[-1]
                smooth_feat /= (np.linalg.norm(smooth_feat) + 1e-6)
                self.features.append(smooth_feat)
            else:
                self.features.append(virtual_feat)

            # Maintain gallery size limit
            if len(self.features) > 10:
                self.features.pop(0)

        # Store virtual box for analysis (only if within reasonable bounds)
        virtual_box = self.primary_kf.x[:4].copy()
        if frame_id is not None:
            self.virtual_boxes.append((frame_id, virtual_box))

    def camera_update(self, warp_matrix):
        """
        Update track state after camera motion compensation.
        State format: [x, y, a, h] where a = aspect ratio, h = height
        """
        M = _build_3x3_warp(warp_matrix)
        x1, y1, x2, y2 = self.to_tlbr()
        x1f = _ensure_scalar(x1); y1f = _ensure_scalar(y1)
        x2f = _ensure_scalar(x2); y2f = _ensure_scalar(y2)

        # Transform corners
        p1 = np.array([x1f, y1f, 1.0], dtype=np.float32).reshape(3, 1)
        p2 = np.array([x2f, y2f, 1.0], dtype=np.float32).reshape(3, 1)
        p1_t = M @ p1
        p2_t = M @ p2
        x1_, y1_ = float(p1_t[0, 0]), float(p1_t[1, 0])
        x2_, y2_ = float(p2_t[0, 0]), float(p2_t[1, 0])

        # Calculate transformed box parameters
        w = abs(x2_ - x1_)
        h = abs(y2_ - y1_)
        cx = (x1_ + x2_) / 2.0
        cy = (y1_ + y2_) / 2.0
        a = (w / h) if (h > 1e-6) else 1.0  # aspect ratio

        # UTRTrack uses [x, y, a, h] format (NOT [cx, cy, s, r])
        # Update both Source and Primary trackers
        new_state = np.array([cx, cy, a, h], dtype=np.float32)

        # Validate new state to prevent overflow/underflow
        if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
            # Skip update if invalid values detected
            return

        if h < 1.0 or h > 10000.0 or w < 1.0 or w > 10000.0:
            # Skip update if box dimensions are unrealistic
            return

        # Update both trackers
        self.source_kf.x[:4] = new_state
        self.primary_kf.x[:4] = new_state

        # Update main state from primary
        self.kf = self.primary_kf
        self.mean = self.primary_kf.x.copy()
        self.covariance = self.primary_kf.P.copy()

    def is_confirmed(self):
        return self.state == 2

    def is_deleted(self):
        return self.time_since_update > self._max_age
