# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
EndoOcSort: OcSort tracker với XYSR-TLUKF motion model.
Giữ nguyên logic OcSort gốc, chỉ thay KalmanFilter thành XYSR-TLUKF.
"""

from collections import deque
import numpy as np

from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.ocsort.ocsort import OcSort, convert_x_to_bbox, speed_direction
from boxmot.motion.kalman_filters.aabb.xysr_tlukf import KalmanFilterXYSR_TLUKF
from boxmot.utils.ops import xyxy2xysr
from boxmot.utils import logger as LOGGER


class KalmanBoxTrackerTLUKF(object):
    """
    OcSort KalmanBoxTracker nhưng sử dụng XYSR-TLUKF thay vì XYSR-KF.
    Giữ nguyên interface của KalmanBoxTracker gốc.
    """

    count = 0

    def __init__(
        self,
        bbox,
        cls,
        det_ind,
        delta_t=3,
        max_obs=50,
        is_source=False,
    ):
        """
        Khởi tạo tracker với XYSR-TLUKF.
        
        Args:
            bbox: [x1, y1, x2, y2, conf]
            cls: Class ID
            det_ind: Detection index
            delta_t: Observation history window (không dùng trong TLUKF nhưng giữ để compatible)
            max_obs: Maximum observations
            is_source: True nếu Source tracker, False nếu Primary
        """
        self.det_ind = det_ind
        self.delta_t = delta_t
        self.max_obs = max_obs

        # Sử dụng XYSR-TLUKF thay vì XYSR-KF
        self.kf = KalmanFilterXYSR_TLUKF(is_source=is_source, max_obs=max_obs)
        
        # Khởi tạo state từ bbox
        bbox_xysr = xyxy2xysr(bbox[:4])
        self.kf.initiate(bbox_xysr)
        
        # OcSort attributes
        self.time_since_update = 0
        self.id = KalmanBoxTrackerTLUKF.count
        KalmanBoxTrackerTLUKF.count += 1
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        
        # OcSort observation tracking
        self.last_observation = np.array(bbox[:5]) if len(bbox) >= 5 else np.r_[bbox, [1.0]]
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        Compatible với OcSort interface.
        """
        self.det_ind = det_ind
        
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            
            # Compute velocity (OcSort style)
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            # Update observations
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            # TLUKF update
            bbox_xysr = xyxy2xysr(bbox[:4])
            self.kf.update(measurement=bbox_xysr, confidence=self.conf)
            
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
        else:
            # No measurement - TLUKF can handle this
            self.kf.update(measurement=None, confidence=None)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # TLUKF predict
        self.kf.predict()
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Convert XYSR state to XYXY for OcSort compatibility
        xysr_state = self.kf.x[:4].flatten()
        xyxy_bbox = self._xysr_to_xyxy(xysr_state)
        
        self.history.append(xyxy_bbox.reshape(1, 4))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate in XYXY format.
        """
        xysr_state = self.kf.x[:4].flatten()
        xyxy_bbox = self._xysr_to_xyxy(xysr_state)
        return xyxy_bbox.reshape(1, 4)
    
    def _xysr_to_xyxy(self, xysr):
        """Convert XYSR to XYXY format."""
        x, y, s, r = xysr
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.array([x1, y1, x2, y2])


class EndoOcSort(OcSort):
    """
    EndoOcSort: OcSort với XYSR-TLUKF motion model.
    
    Kế thừa toàn bộ logic từ OcSort, chỉ thay KalmanBoxTracker bằng KalmanBoxTrackerTLUKF.
    """
    
    def __init__(
        self,
        # BaseTracker parameters
        det_thresh: float = 0.2,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        # OcSort-specific parameters
        min_conf: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
        # TLUKF-specific (không dùng Q_xy_scaling, Q_s_scaling vì TLUKF tự quản lý)
        is_source: bool = False,  # Source hay Primary tracker
        **kwargs
    ):
        # Gọi OcSort.__init__ nhưng không dùng Q_xy_scaling, Q_s_scaling
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
            min_conf=min_conf,
            delta_t=delta_t,
            inertia=inertia,
            use_byte=use_byte,
            Q_xy_scaling=0.01,  # Dummy values, won't be used
            Q_s_scaling=0.0001,
            **kwargs
        )
        
        # TLUKF parameter
        self.is_source = is_source
        
        # Reset counter cho KalmanBoxTrackerTLUKF
        KalmanBoxTrackerTLUKF.count = 0
        
        LOGGER.success("Initialized EndoOcSort with XYSR-TLUKF")
    
    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Override OcSort.update để sử dụng KalmanBoxTrackerTLUKF.
        Giữ nguyên toàn bộ logic của OcSort, chỉ thay tracker class.
        """
        self.check_inputs(dets, img)

        self.frame_count += 1
        h, w = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4 + self.is_obb]

        inds_low = confs > self.min_conf
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.active_tracks), 5 + self.is_obb))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[i] for i in range(4 + self.is_obb)] + [0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array([
            trk.velocity if trk.velocity is not None else np.array((0, 0))
            for trk in self.active_tracks
        ])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])

        from boxmot.trackers.ocsort.ocsort import k_previous_obs
        k_observations = np.array([
            k_previous_obs(trk.observations, trk.age, self.delta_t, is_obb=self.is_obb)
            for trk in self.active_tracks
        ])

        # First round of association
        from boxmot.utils.association import associate
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0 : 5 + self.is_obb],
            trks,
            self.asso_func,
            self.asso_threshold,
            velocities,
            k_observations,
            self.inertia,
            w,
            h,
        )
        
        for m in matched:
            self.active_tracks[m[1]].update(
                dets[m[0], :-2], dets[m[0], -2], dets[m[0], -1]
            )

        # Second round of association by OCR (BYTE)
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            
            if iou_left.max() > self.asso_threshold:
                from boxmot.utils.association import linear_assignment
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :-2],
                        dets_second[det_ind, -2],
                        dets_second[det_ind, -1],
                    )
                    to_remove_trk_indices.append(trk_ind)
                
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Third round with observation history
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            
            if iou_left.max() > self.asso_threshold:
                from boxmot.utils.association import linear_assignment
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets[det_ind, :-2], dets[det_ind, -2], dets[det_ind, -1]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Update unmatched tracks (no measurement)
        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # Create new trackers - SỬ DỤNG KalmanBoxTrackerTLUKF thay vì KalmanBoxTracker
        for i in unmatched_dets:
            if self.is_obb:
                # OBB not supported with TLUKF yet
                raise NotImplementedError("OBB not supported with TLUKF")
            else:
                trk = KalmanBoxTrackerTLUKF(
                    dets[i, :5],
                    dets[i, 5],
                    dets[i, 6],
                    delta_t=self.delta_t,
                    max_obs=self.max_obs,
                    is_source=self.is_source,
                )
            self.active_tracks.append(trk)
        
        # Return active tracks
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[: 4 + self.is_obb]
            
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate(
                        (d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])
                    ).reshape(1, -1)
                )
            i -= 1
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 8))
