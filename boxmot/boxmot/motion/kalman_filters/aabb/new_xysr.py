import numpy as np
from numpy import dot, eye
from typing import Tuple
from copy import deepcopy
import warnings


class KalmanFilterXYSR:
    """
    Kalman Filter with state: [x, y, s, r, vx, vy, vs, vr]^T
    where
        x, y : center
        s    : scale (area)
        r    : aspect ratio (w/h)
        v*   : corresponding velocities
    """

    ndim = 4
    dt = 1.0
    _eps = 1e-7
    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    def __init__(self):
        # State transition F
        self.F = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = self.dt

        # Measurement matrix H
        self.H = np.eye(self.ndim, 2 * self.ndim)

        # Initial state, covariance
        self.x = np.zeros((2 * self.ndim, 1))
        self.P = np.eye(2 * self.ndim)

        # Prior storage
        self.x_prior = None
        self.P_prior = None

        # History
        self.history_obs = []
        self.history_pred = []
        self.last_measurement = None
        self.observed = True
        self.frozen = False
        self.attr_saved = None

    # -------------------------------------------------------------
    # Covariance std definitions
    # -------------------------------------------------------------
    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        return [
            2 * self._std_weight_position * measurement[2],     # x
            2 * self._std_weight_position * measurement[2],     # y
            1e-2,                                               # s
            2 * self._std_weight_position * measurement[2],     # r
            10 * self._std_weight_velocity * measurement[2],    # vx
            10 * self._std_weight_velocity * measurement[2],    # vy
            1e-5,                                               # vs
            10 * self._std_weight_velocity * measurement[2],    # vr
        ]

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-2,
            self._std_weight_position * mean[2],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[2],
            1e-5,
            self._std_weight_velocity * mean[2],
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        conf = np.clip(confidence, 0.1, 1.0)  # tránh 0
        std_noise = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-1,
            self._std_weight_position * mean[2],
        ]
        return np.array(std_noise) / conf

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.x = np.r_[mean_pos, mean_vel].reshape(-1, 1)
        std = self._get_initial_covariance_std(measurement)
        self.P = np.diag(np.square(std))
        return self.x.copy(), self.P.copy()

    # -------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        std_pos, std_vel = self._get_process_noise_std(self.x.flatten())
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.x = dot(self.F, self.x)
        self.P = dot(self.F, self.P).dot(self.F.T) + Q

        # enforce symmetry + jitter
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.P.shape[0]) * 1e-9

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.history_pred.append(self.x[:4].flatten())
        return self.x.copy(), self.P.copy()

    # -------------------------------------------------------------
    # Update
    # -------------------------------------------------------------
    def update(self, measurement: np.ndarray, confidence: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if measurement is None:
            self.last_measurement = self.x[:4].copy()
            self.observed = False
            self.history_obs.append(None)
            return self.x.copy(), self.P.copy()

        self.observed = True
        z = measurement.reshape(-1, 1)

        std = self._get_measurement_noise_std(self.x.flatten(), confidence)
        R = np.diag(np.square(std))

        # Innovation
        y = z - dot(self.H, self.x)
        PHT = dot(self.P, self.H.T)
        S = dot(self.H, PHT) + R

        try:
            K = np.linalg.solve(S.T, PHT.T).T  # stable than inv
        except np.linalg.LinAlgError:
            warnings.warn("S not SPD, using pinv")
            K = PHT @ np.linalg.pinv(S)

        self.x = self.x + dot(K, y)
        I = np.eye(self.F.shape[0])
        self.P = (I - dot(K, self.H)) @ self.P @ (I - dot(K, self.H)).T + K @ R @ K.T

        # enforce symmetry + jitter
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.P.shape[0]) * 1e-9

        self.history_obs.append(z.flatten())
        return self.x.copy(), self.P.copy()

    # -------------------------------------------------------------
    # Project measurement distribution
    # -------------------------------------------------------------
    def project(self, mean: np.ndarray, covariance: np.ndarray):
        mean = dot(self.H, mean)
        covariance = dot(self.H, covariance).dot(self.H.T)
        return mean, covariance

    # -------------------------------------------------------------
    # Gating distance (Mahalanobis with noise R)
    # -------------------------------------------------------------
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean, projected_cov = projected_mean[:2], projected_cov[:2, :2]
            measurements = measurements[:, :2]

        R = np.diag(np.square(self._get_measurement_noise_std(mean.flatten(), 1.0)))
        S = projected_cov + R + np.eye(projected_cov.shape[0]) * 1e-6

        try:
            chol = np.linalg.cholesky(S)
            d = measurements - projected_mean.reshape(1, -1)
            z = np.linalg.solve(chol, d.T)
            squared_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            d = measurements - projected_mean.reshape(1, -1)
            squared_maha = np.sum(d @ S_inv * d, axis=1)
        return squared_maha

    # -------------------------------------------------------------
    # Logpdf likelihood
    # -------------------------------------------------------------
    @staticmethod
    def logpdf(x, mean, cov):
        cov = cov + np.eye(cov.shape[0]) * 1e-9
        diff = x - mean
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            sign, logdet = np.linalg.slogdet(cov)
        sol = np.linalg.solve(cov, diff)
        mahal = float(diff.T @ sol)
        dim = x.shape[0]
        return -0.5 * (dim * np.log(2 * np.pi) + logdet + mahal)

    # -------------------------------------------------------------
    # Convert state -> bbox
    # -------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        ret = self.x[:4].flatten()
        return ret

    def get_bbox(self) -> np.ndarray:
        cx, cy, s, r = self.x[:4].flatten()
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        return np.array([cx, cy, w, h])

    # -------------------------------------------------------------
    # Affine correction (A P A^T)
    # -------------------------------------------------------------
    def apply_affine_correction(self, M: np.ndarray):
        """
        Apply 2x3 affine matrix M to x,y and vx,vy
        """
        A = np.eye(8)
        A[0:2, 0:2] = M[:2, :2]
        A[4:6, 4:6] = M[:2, :2]
        self.x = A @ self.x
        self.P = A @ self.P @ A.T
        self.P = 0.5 * (self.P + self.P.T) + np.eye(8) * 1e-9
