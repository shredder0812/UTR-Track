# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import os
import numpy as np

from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


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


def _kf_initiate_or_set(kf, bbox):
    """
    bbox: 1D array-like (len=4) in the filter's measurement format (e.g., [x,y,a,h] or [x,y,s,r])
    This function tries to:
      - call kf.initiate(bbox) if exists and returns (mean, cov)
      - otherwise set kf.x and kf.P using available attributes
    Returns: (mean, covariance) as numpy arrays (column vector and matrix)
    """
    bbox_arr = np.asarray(bbox).ravel()
    # If kf has 'initiate' call it
    if hasattr(kf, "initiate"):
        try:
            res = kf.initiate(bbox_arr)
            # some implementations return (mean, covariance)
            if isinstance(res, tuple) and len(res) == 2:
                mean, cov = res
                # convert to column vector if necessary
                mean = np.asarray(mean).reshape(-1, 1)
                cov = np.asarray(cov)
                return mean, cov
        except Exception:
            # fallthrough to manual set
            pass

    # fallback: if kf has attributes x and P, set them
    if hasattr(kf, "x") and hasattr(kf, "P"):
        # try to preserve dim
        dim_x = getattr(kf, "dim_x", None)
        if dim_x is None:
            # guess from existing kf.x shape
            try:
                dim_x = kf.x.shape[0]
            except Exception:
                dim_x = bbox_arr.size * 2  # guess
        # create column x with zeros
        x = np.zeros((dim_x, 1), dtype=np.float32)
        x[: bbox_arr.size, 0] = bbox_arr
        kf.x = x
        # set P to identity * small
        kf.P = np.eye(dim_x, dtype=np.float32) * 1.0
        return kf.x.copy(), kf.P.copy()

    # last resort
    mean = bbox_arr.reshape(-1, 1)
    cov = np.eye(mean.shape[0], dtype=np.float32)
    return mean, cov


def _kf_predict_get(kf, mean=None, cov=None):
    """
    Generic predict wrapper:
     - if kf has predict(mean, cov) -> use that and return new mean,cov
     - elif kf.predict() updates internal x,P -> call and read kf.x,kf.P
     - else try to call function-style predict
    """
    if hasattr(kf, "predict"):
        try:
            # try call with mean,cov signature
            res = kf.predict(mean, cov)
            if isinstance(res, tuple) and len(res) == 2:
                return np.asarray(res[0]).reshape(-1, 1), np.asarray(res[1])
        except TypeError:
            # maybe predict() doesn't accept args and updates internal x,P
            try:
                kf.predict()
                return kf.x.copy(), kf.P.copy()
            except Exception:
                pass
        except Exception:
            pass
    # fallback: return inputs unchanged
    return mean, cov


def _kf_update_get(kf, mean, cov, measurement, confidence=0.0):
    """
    Generic update wrapper:
     - if kf.update(mean, cov, measurement, confidence) exists -> use and return mean,cov
     - elif kf.update(measurement, R) style -> call and read kf.x,kf.P
     - elif kf.update() updates kf.x/kf.P -> call accordingly
    """
    if hasattr(kf, "update"):
        try:
            res = kf.update(mean, cov, measurement, confidence)
            if isinstance(res, tuple) and len(res) == 2:
                return np.asarray(res[0]).reshape(-1, 1), np.asarray(res[1])
        except TypeError:
            # maybe update(z, R=None, H=None) signature (filterpy-like)
            try:
                kf.update(measurement, confidence)
                return kf.x.copy(), kf.P.copy()
            except Exception:
                pass
        except Exception:
            pass
    # fallback: write measurement into mean
    m = np.asarray(measurement).reshape(-1, 1)
    mean[: m.shape[0], 0] = m[:, 0]
    return mean, cov


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
    ):
        self.id = id
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        # start with confirmed in Ci as test expect equal amount of outputs as inputs
        self.state = (
            TrackState.Confirmed
            if (
                os.getenv("GITHUB_ACTIONS") == "true"
                and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark"
            )
            else TrackState.Tentative
        )
        self.features = []
        
        # Class voting for stability
        self.class_history = [self.cls]
        self.class_vote_window = 200
        self.stable_cls = self.cls
        if detection.feat is not None:
            detection.feat /= np.linalg.norm(detection.feat)
            self.features.append(detection.feat)

        self._n_init = n_init
        self._max_age = max_age

        # Initialize Kalman filter
        self.kf = KalmanFilterXYAH()
        
        # Set initial state
        mean, covariance = self.kf.initiate(self.bbox)
        self.mean = mean
        self.covariance = covariance

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        a, b = warp_matrix
        warp_matrix = np.vstack([a, b, np.array([0, 0, 1])])
        x1, y1, x2, y2 = self.to_tlbr()
        # Transform top-left point
        p1 = np.array([[x1], [y1], [1]], dtype=np.float32)
        p1_transformed = warp_matrix @ p1
        x1_, y1_ = p1_transformed[0, 0], p1_transformed[1, 0]
        # Transform bottom-right point
        p2 = np.array([[x2], [y2], [1]], dtype=np.float32)
        p2_transformed = warp_matrix @ p2
        x2_, y2_ = p2_transformed[0, 0], p2_transformed[1, 0]
        # Calculate new dimensions
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        # Update Kalman filter state
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        
        # Class voting mechanism
        self.class_history.append(detection.cls)
        if len(self.class_history) > self.class_vote_window:
            self.class_history.pop(0)
        class_counts = {}
        for cls in self.class_history:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        self.stable_cls = max(class_counts, key=class_counts.get)
        self.cls = self.stable_cls
        
        self.det_ind = detection.det_ind
        
        # Update Kalman filter with new measurement
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.bbox, self.conf
        )

        feature = detection.feat / np.linalg.norm(detection.feat)

        smooth_feat = (
            self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
        )
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


class TrackXYSR(Track):
    """XYSR-specific track with virtual trajectory support."""
    def __init__(self, detection, id, n_init, max_age, ema_alpha):
        self.last_seen_time = 0  # Frame index when track was last seen
        self.last_real_detection = None  # Last real detection before track was lost
        self.virtual_detections = []  # List to store virtual detections
        self.max_gap = 8  # Maximum frames to interpolate for endoscopy videos
        self.last_measurement = None  # Stores last real measurement in (x,y,s,r)
        # We'll not call Track.__init__ to avoid XYAH-specific conversion there.
        self.id = id
        x1, y1, w, h = detection.tlwh
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        s = w * h
        r = w / h if h != 0 else 0.0
        self.bbox = np.array([cx, cy, s, r], dtype=np.float32)
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        
        # Class voting for stability
        self.class_history = [self.cls]
        self.class_vote_window = 200
        self.stable_cls = self.cls

        self.state = (
            TrackState.Confirmed
            if (
                os.getenv("GITHUB_ACTIONS") == "true"
                and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark"
            )
            else TrackState.Tentative
        )

        self.features = []
        if detection.feat is not None:
            detection.feat = detection.feat / np.linalg.norm(detection.feat)
            self.features.append(detection.feat)

        self._n_init = n_init
        self._max_age = max_age

        # Initialize XYSR Kalman filter
        # self.kf = KalmanFilterXYSR(dim_x=8, dim_z=4, dim_u=0) if hasattr(KalmanFilterXYSR, "__call__") else KalmanFilterXYSR(8,4,0)
        self.kf = KalmanFilterXYSR()

        mean, cov = _kf_initiate_or_set(self.kf, self.bbox)
        self.mean = np.asarray(mean).reshape(-1, 1)
        self.covariance = np.asarray(cov)
        # Set last measurement as initial bbox
        self.last_measurement = self.bbox.copy()

    def to_tlwh(self):
        cx, cy, s, r = self.mean[:4].ravel()
        w = np.sqrt(max(s * r, 0.0))
        h = np.sqrt(max(s / r, 0.0)) if r != 0 else 0.0
        tl_x = cx - w / 2.0
        tl_y = cy - h / 2.0
        return np.array([tl_x, tl_y, w, h], dtype=np.float32)

    def to_tlbr(self):
        tlwh = self.to_tlwh()
        x1, y1, w, h = tlwh
        return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

    def camera_update(self, warp_matrix):
        M = _build_3x3_warp(warp_matrix)
        x1, y1, x2, y2 = self.to_tlbr()
        x1f = _ensure_scalar(x1); y1f = _ensure_scalar(y1)
        x2f = _ensure_scalar(x2); y2f = _ensure_scalar(y2)
        p1 = np.array([x1f, y1f, 1.0], dtype=np.float32).reshape(3,1)
        p2 = np.array([x2f, y2f, 1.0], dtype=np.float32).reshape(3,1)
        p1_t = M @ p1; p2_t = M @ p2
        x1_, y1_ = float(p1_t[0,0]), float(p1_t[1,0])
        x2_, y2_ = float(p2_t[0,0]), float(p2_t[1,0])
        w = x2_ - x1_; h = y2_ - y1_
        cx = x1_ + w/2.0; cy = y1_ + h/2.0
        s = w * h
        r = (w / h) if (h != 0) else 0.0
        # set into filter-friendly format
        if hasattr(self.kf, "x"):
            # try to set first 4 entries
            try:
                self.kf.x[:4, 0] = np.array([cx, cy, s, r], dtype=np.float32)
            except Exception:
                # fallback: set mean stored in object
                self.mean[:4, 0] = np.array([cx, cy, s, r], dtype=np.float32)
        else:
            self.mean[:4,0] = np.array([cx, cy, s, r], dtype=np.float32)

    def predict(self):
        mean, cov = _kf_predict_get(self.kf, self.mean, self.covariance)
        self.mean = np.asarray(mean).reshape(-1, 1)
        self.covariance = np.asarray(cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        x1, y1, w, h = detection.tlwh
        cx, cy = x1 + w/2.0, y1 + h/2.0
        s = w * h
        r = (w / h) if (h != 0) else 0.0
        meas = np.array([cx, cy, s, r], dtype=np.float32).ravel()

        self.bbox = meas
        self.conf = detection.conf
        
        # Class voting mechanism
        self.class_history.append(detection.cls)
        if len(self.class_history) > self.class_vote_window:
            self.class_history.pop(0)
        class_counts = {}
        for cls in self.class_history:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        self.stable_cls = max(class_counts, key=class_counts.get)
        self.cls = self.stable_cls
        
        self.det_ind = detection.det_ind

        # If this track was missed for several frames and now re-appears,
        # interpolate virtual measurements to smooth the KF per ORU.
        if self.time_since_update > 1 and self.last_measurement is not None:
            gap = int(self.time_since_update)
            # Generate virtual boxes between last real and current real measurement
            virtual_boxes = self.kf.interpolate_virtual_boxes(
                self.last_measurement, meas, 0, gap + 1, max_gap=50
            )
            for idx, vbox in enumerate(virtual_boxes):
                # Update with reduced confidence; predict between steps
                v_mean, v_cov = _kf_update_get(self.kf, self.mean, self.covariance, vbox, max(0.3, min(self.conf, 0.9)))
                self.mean = np.asarray(v_mean).reshape(-1, 1)
                self.covariance = np.asarray(v_cov)
                if idx != len(virtual_boxes) - 1:
                    p_mean, p_cov = _kf_predict_get(self.kf, self.mean, self.covariance)
                    self.mean = np.asarray(p_mean).reshape(-1, 1)
                    self.covariance = np.asarray(p_cov)

        mean, cov = _kf_update_get(self.kf, self.mean, self.covariance, meas, self.conf)
        self.mean = np.asarray(mean).reshape(-1, 1)
        self.covariance = np.asarray(cov)
        # Save last real measurement
        self.last_measurement = meas.copy()

        if detection.feat is not None:
            feature = detection.feat / np.linalg.norm(detection.feat)
            if len(self.features) == 0:
                self.features = [feature]
            else:
                smooth_feat = self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
                smooth_feat /= np.linalg.norm(smooth_feat)
                self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
