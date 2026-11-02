# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
"""
XYSR Transfer-Learning Unscented Kalman Filter used by UTR-Track-v1.

State  : [x, y, s, r, vx, vy, vs, vr]   (centre, area, aspect ratio w/h, velocities)
Measure: [x, y, s, r]

Two instances run per track: a *Source* filter (updated only by high-confidence
detections) and a *Primary* filter (updated by every matched detection and, when
the track is unmatched, by the Source prediction used as a virtual measurement).

Difference from the UTR-Track-v0 filter (``tlukf.py``, XYAH state):

* XYSR state instead of XYAH.
* Process noise ``Q`` calibrated from ground-truth accelerations
  (``q_calibration_scale``, v1 default ``1.0``). On 5,911 ground-truth steps of the
  18 IGH sequences the hand-set ``Q`` gave mean NIS 42.5 (theory: 4) and rejected
  40.8% of correct measurements at the chi^2(4) 95% gate; the calibrated ``Q``
  (m = 1) gives mean NIS 12.05 and a 6.0% rejection rate. ``q_calibration_scale=None``
  restores the hand-set ``Q`` (ablation only).
* The transfer-learning step uses the Source covariance ``P_eta`` as the noise of the
  virtual measurement, so a confident Source pulls Primary harder than an uncertain one.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

# Standard deviation of ground-truth second differences (IGH, 18 sequences):
# position [px], area [px^2], aspect ratio [-].
_ACC_STD_POS = 18.7421
_ACC_STD_AREA = 25294.6
_ACC_STD_RATIO = 0.02


class KalmanFilterXYSRTLUKFv1:
    """Unscented Kalman filter in XYSR space with calibrated process noise.

    Args:
        is_source: ``True`` for the Source (teacher) filter, which only accepts real
            measurements; ``False`` for the Primary (student) filter, which can also
            learn from the Source prediction.
        q_calibration_scale: scale ``m`` of the calibrated process noise
            ``Q = m * diag(sigma_acc^2)``. ``None`` keeps the original hand-set ``Q``.
    """

    def __init__(self, is_source: bool = False, q_calibration_scale: float | None = 1.0):
        self.nx = 8  # [x, y, s, r, vx, vy, vs, vr]
        self.nz = 4  # [x, y, s, r]

        # Unscented transform parameters
        self.alpha = 1e-1
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self.dt = 1.0
        self.is_source = is_source

        # Numerical stability constants
        self._PSD_MIN_EIG = 1e-3
        self._CHOLESKY_REG_FACTOR = 1e-4

        self.q_calibration_scale = q_calibration_scale
        if q_calibration_scale is not None:
            m = float(q_calibration_scale)
            sa_pos, sa_s, sa_r = _ACC_STD_POS, _ACC_STD_AREA, _ACC_STD_RATIO
            self.Q = np.diag([
                m * sa_pos ** 2, m * sa_pos ** 2,   # x, y
                m * sa_s ** 2, m * sa_r ** 2,       # s, r
                m * sa_pos ** 2, m * sa_pos ** 2,   # vx, vy
                m * sa_s ** 2, m * sa_r ** 2,       # vs, vr
            ]) * self.dt
        else:
            # Original hand-set process noise (UTR-Track-v0 values).
            self.Q = np.diag([
                0.5, 0.5,      # x, y
                1e-6, 1e-6,    # s, r
                1.0, 1.0,      # vx, vy
                1e-8, 1e-8,    # vs, vr
            ]) * self.dt

        # Measurement noise R (re-computed from detection confidence at every update)
        std = [
            self._std_weight_position,
            self._std_weight_position,
            0.1,
            self._std_weight_position,
        ]
        self.R = np.diag(np.square(std))

        self.x = np.zeros(self.nx)
        self.P = np.eye(self.nx)
        self.Wm, self.Wc = self._compute_weights()

    # ------------------------------------------------------------------ helpers
    def _compute_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the unscented-transform mean and covariance weights."""
        Wm = np.full(2 * self.nx + 1, 0.5 / (self.nx + self.lambda_))
        Wc = Wm.copy()
        Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        Wc[0] = Wm[0] + (1 - self.alpha**2 + self.beta)
        return Wm, Wc

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Symmetrise, regularise and, if needed, eigen-clip a covariance matrix."""
        matrix = 0.5 * (matrix + matrix.T)
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return np.eye(matrix.shape[0]) * 1.0
        matrix += np.eye(matrix.shape[0]) * self._CHOLESKY_REG_FACTOR
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, self._PSD_MIN_EIG)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate the 2n+1 sigma points of N(x, P)."""
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.zeros_like(x)
            P = np.eye(len(x))
        P_stable = self._ensure_positive_definite(P)
        try:
            L = np.linalg.cholesky(P_stable)
        except (np.linalg.LinAlgError, ValueError):
            L = np.eye(self.nx) * 0.1

        sigma_points = np.zeros((2 * self.nx + 1, self.nx))
        sigma_points[0] = x
        scale = np.sqrt(self.nx + self.lambda_)
        for i in range(self.nx):
            sigma_points[i + 1] = x + scale * L[:, i]
            sigma_points[self.nx + i + 1] = x - scale * L[:, i]
        return sigma_points

    def _motion_model(self, sigma_points: np.ndarray) -> np.ndarray:
        """Constant-velocity transition applied to every sigma point."""
        transitioned = np.copy(sigma_points)
        transitioned[:, :4] += self.dt * sigma_points[:, 4:]
        return transitioned

    def _measurement_model(self, sigma_points: np.ndarray) -> np.ndarray:
        """Observe [x, y, s, r]."""
        return sigma_points[:, :self.nz]

    # --------------------------------------------------------------- filtering
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialise the state from a first XYSR measurement (zero velocity)."""
        measurement = np.asarray(measurement).flatten()
        self.x = np.r_[measurement[:self.nz], np.zeros(self.nz)]
        s = measurement[2]
        std = [
            2 * self._std_weight_position * np.sqrt(s),
            2 * self._std_weight_position * np.sqrt(s),
            2 * self._std_weight_position * s,
            1e-2,
            10 * self._std_weight_velocity * np.sqrt(s),
            10 * self._std_weight_velocity * np.sqrt(s),
            10 * self._std_weight_velocity * s,
            1e-5,
        ]
        self.P = np.diag(np.square(std))
        return self.x, self.P

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Unscented prediction step."""
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_pred = self._motion_model(sigma_points)
        mean_pred = np.dot(self.Wm, sigma_pred)
        diff = sigma_pred - mean_pred
        cov_pred = np.dot(diff.T * self.Wc, diff)
        cov_pred += self.Q
        self.x = mean_pred
        self.P = self._ensure_positive_definite(cov_pred)
        return self.x, self.P

    def _update_noise(self, measurement: np.ndarray, confidence: float | None) -> None:
        """Set R from the detection confidence: lower confidence -> larger noise."""
        s = float(measurement[2])
        conf_value = float(confidence) if confidence is not None else 0.0
        uncertainty_factor = 1.0 - np.clip(conf_value, 0.0, 1.0)
        pos_std = self._std_weight_position * np.sqrt(s) * (1.0 + uncertainty_factor)
        ar_std = 0.1 * (1.0 + uncertainty_factor)
        std = np.array([
            max(pos_std, 0.01),
            max(pos_std, 0.01),
            max(pos_std * np.sqrt(s), 0.01),
            max(ar_std, 0.01),
        ])
        self.R = np.diag(np.square(std))

    def _perform_single_update(self, measurement: np.ndarray) -> None:
        """One unscented measurement update with the current ``self.R``."""
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_meas = self._measurement_model(sigma_points)
        z_pred = np.dot(self.Wm, sigma_meas)
        diff_z = sigma_meas - z_pred
        Pzz = np.dot(diff_z.T * self.Wc, diff_z) + self.R
        diff_x = sigma_points - self.x
        Pxz = np.dot(diff_x.T * self.Wc, diff_z)
        try:
            K = np.dot(Pxz, inv(Pzz))
        except np.linalg.LinAlgError:
            K = np.zeros((self.nx, self.nz))
        innovation = measurement - z_pred
        self.x += np.dot(K, innovation)
        self.P -= np.dot(K, np.dot(Pzz, K.T))

    def update(
        self,
        measurement: np.ndarray | None = None,
        confidence: float | None = None,
        eta_pred: np.ndarray | None = None,
        P_eta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measurement update and/or transfer-learning update.

        Args:
            measurement: real XYSR measurement, or ``None`` when the track is unmatched.
            confidence: detection confidence of ``measurement``.
            eta_pred: Source state, used by the Primary filter as a virtual measurement
                (only its first four components are used).
            P_eta: Source covariance; its 4x4 position block is the virtual measurement noise.
        """
        if measurement is not None:
            measurement = np.asarray(measurement).flatten()
            self._update_noise(measurement, confidence)

        if self.is_source:
            if measurement is not None:
                self._perform_single_update(measurement)
        else:
            if measurement is not None:
                self._perform_single_update(measurement)
            if eta_pred is not None and P_eta is not None:
                eta_pred = np.asarray(eta_pred).flatten()[: self.nz]
                P_eta_sub = np.asarray(P_eta)[: self.nz, : self.nz]
                valid_eta = not (np.any(np.isnan(eta_pred)) or np.any(np.isinf(eta_pred)))
                valid_P = not (np.any(np.isnan(P_eta_sub)) or np.any(np.isinf(P_eta_sub)))
                if valid_eta and valid_P:
                    original_R = self.R.copy()
                    self.R = self._ensure_positive_definite(P_eta_sub)
                    self._perform_single_update(eta_pred)
                    self.R = original_R

        self.P = self._ensure_positive_definite(self.P)
        return self.x, self.P

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Squared Mahalanobis distance between the predicted measurement and ``measurements``."""
        mean = np.asarray(mean).flatten()
        measurements = np.asarray(measurements)
        if len(measurements) == 0:
            return np.array([])
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)

        if only_position:
            mean_meas = mean[:2]
            cov_meas = covariance[:2, :2]
            measurements = measurements[:, :2]
        else:
            mean_meas = mean[:self.nz]
            cov_meas = covariance[:self.nz, :self.nz]
            measurements = measurements[:, :self.nz]

        cov_meas = cov_meas + self.R[:len(mean_meas), :len(mean_meas)]
        cov_meas = self._ensure_positive_definite(cov_meas)
        try:
            cholesky_factor = np.linalg.cholesky(cov_meas)
            d = measurements - mean_meas
            z = np.linalg.solve(cholesky_factor, d.T).T
            squared_maha = np.sum(z * z, axis=1)
            if metric in ("maha", "gaussian"):
                return squared_maha
            raise ValueError(f"Invalid metric: {metric}")
        except np.linalg.LinAlgError:
            return np.sum((measurements - mean_meas) ** 2, axis=1)
