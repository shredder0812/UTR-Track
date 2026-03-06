"""
XYSR-TLUKF Kalman Filter:
Kết hợp state space XYSR (x, y, scale, ratio) với Transfer Learning UKF.
Được thiết kế đặc biệt cho endoscopy tracking với khả năng học từ source tracker.
"""

import numpy as np
from numpy.linalg import inv
from collections import deque

# Chi-squared threshold cho gating ở mức tin cậy 95%
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

def xyxy2xysr(bbox_xyxy):
    """
    Chuyển đổi bbox từ [x1, y1, x2, y2] sang [x, y, s, r]
    x, y: tâm bbox
    s: diện tích (scale)
    r: tỷ lệ w/h (aspect ratio)
    """
    x1, y1, x2, y2 = bbox_xyxy[:4]
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    s = w * h  # scale (area)
    r = w / (h + 1e-6)  # aspect ratio
    result = np.array([x, y, s, r])
    if len(bbox_xyxy) > 4:
        return np.r_[result, bbox_xyxy[4:]]
    return result

def xysr2xyxy(bbox_xysr):
    """
    Chuyển đổi bbox từ [x, y, s, r] sang [x1, y1, x2, y2]
    """
    x, y, s, r = bbox_xysr[:4]
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    result = np.array([x1, y1, x2, y2])
    if len(bbox_xysr) > 4:
        return np.r_[result, bbox_xysr[4:]]
    return result


class KalmanFilterXYSR_TLUKF:
    """
    XYSR-TLUKF: Unscented Kalman Filter cho tracking với state space XYSR.
    
    State vector: [x, y, s, r, vx, vy, vs, vr] (8 dimensional)
    - x, y: center position
    - s: scale (area = w * h)
    - r: aspect ratio (w / h)
    - vx, vy, vs, vr: velocities
    
    Hỗ trợ Transfer Learning:
    - Source tracker: High-confidence tracker (conf >= 0.6)
    - Primary tracker: Learns from Source during gaps/low-conf periods
    """
    
    def __init__(self, is_source=False, max_obs=50):
        """
        Args:
            is_source: True nếu đây là Source tracker, False nếu là Primary tracker
            max_obs: Maximum số observations lưu trữ
        """
        # Dimensions
        self.nx = 8  # State dimension [x, y, s, r, vx, vy, vs, vr]
        self.nz = 4  # Measurement dimension [x, y, s, r]
        
        # UKF parameters
        self.alpha = 1e-1
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx
        
        # Noise weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self.dt = 1.0
        
        # Transfer Learning flag
        self.is_source = is_source
        
        # Numerical stability constants
        self._PSD_MIN_EIG = 1e-3
        self._CHOLESKY_REG_FACTOR = 1e-4
        
        # Process noise Q - CRITICAL: Very low noise for size velocities
        self.Q = np.diag([
            0.5, 0.5,      # Position noise (x, y)
            1e-6, 1e-6,    # Scale/Ratio noise (s, r) - VERY LOW for endoscopy
            1.0, 1.0,      # Velocity position noise (vx, vy)
            1e-8, 1e-8     # Velocity size noise (vs, vr) - EXTREMELY LOW
        ]) * self.dt
        
        # Measurement noise R (updated dynamically)
        std = [
            self._std_weight_position,
            self._std_weight_position,
            0.1,
            self._std_weight_position
        ]
        self.R = np.diag(np.square(std))
        
        # State
        self.x = np.zeros(self.nx)  # State mean
        self.P = np.eye(self.nx)    # State covariance
        
        # UKF weights
        self.Wm, self.Wc = self._compute_weights()
        
        # History tracking
        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=max_obs)
        self.observed = False
        self.last_measurement = None
        
    def _compute_weights(self):
        """Compute UKF sigma point weights."""
        Wm = np.full(2 * self.nx + 1, 0.5 / (self.nx + self.lambda_))
        Wc = Wm.copy()
        Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        Wc[0] = Wm[0] + (1 - self.alpha**2 + self.beta)
        return Wm, Wc
    
    def _ensure_positive_definite(self, matrix):
        """Ensure covariance matrix is symmetric and positive definite."""
        # Symmetry
        matrix = 0.5 * (matrix + matrix.T)
        
        # Check for NaN/Inf
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return np.eye(matrix.shape[0]) * 1.0
        
        # Add regularization
        matrix += np.eye(matrix.shape[0]) * self._CHOLESKY_REG_FACTOR
        
        try:
            # Test Cholesky decomposition
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # Eigenvalue fix
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, self._PSD_MIN_EIG)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def _generate_sigma_points(self, x, P):
        """Generate sigma points from mean and covariance."""
        # Validate state
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
    
    def _motion_model(self, sigma_points):
        """State transition (constant velocity model)."""
        transitioned = np.copy(sigma_points)
        dt = self.dt
        transitioned[:, :4] += dt * sigma_points[:, 4:]  # position += velocity * dt
        # Velocities remain constant
        return transitioned
    
    def _measurement_model(self, sigma_points):
        """Measurement model (observe x, y, s, r)."""
        return sigma_points[:, :self.nz]
    
    def initiate(self, measurement):
        """Initialize tracker from first measurement in XYSR format."""
        measurement = np.asarray(measurement).flatten()
        
        # Initial state
        mean_pos = measurement[:self.nz]
        mean_vel = np.zeros(self.nz)
        self.x = np.r_[mean_pos, mean_vel]
        
        # Initial covariance
        s = measurement[2]  # scale
        std = [
            2 * self._std_weight_position * np.sqrt(s),
            2 * self._std_weight_position * np.sqrt(s),
            2 * self._std_weight_position * s,
            1e-2,
            10 * self._std_weight_velocity * np.sqrt(s),
            10 * self._std_weight_velocity * np.sqrt(s),
            10 * self._std_weight_velocity * s,
            1e-5
        ]
        self.P = np.diag(np.square(std))
        
        # History
        self.history_obs.append(measurement)
        self.last_measurement = measurement
        self.observed = True
        
        return self.x, self.P
    
    def predict(self):
        """Predict next state using UKF."""
        # Generate and propagate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_pred = self._motion_model(sigma_points)
        
        # Compute predicted mean and covariance
        mean_pred = np.dot(self.Wm, sigma_pred)
        diff = sigma_pred - mean_pred
        cov_pred = np.dot(diff.T * self.Wc, diff)
        cov_pred += self.Q  # Add process noise
        
        self.x = mean_pred
        self.P = self._ensure_positive_definite(cov_pred)
        
        return self.x, self.P
    
    def _update_noise(self, measurement, confidence):
        """Update measurement noise R dynamically based on confidence."""
        s = float(measurement[2])  # scale
        conf_value = float(confidence) if confidence is not None else 0.0
        
        # Uncertainty increases when confidence decreases
        uncertainty_factor = 1.0 - np.clip(conf_value, 0.0, 1.0)
        
        pos_std = self._std_weight_position * np.sqrt(s) * (1.0 + uncertainty_factor)
        ar_std = 0.1 * (1.0 + uncertainty_factor)
        
        std = np.array([
            max(pos_std, 0.01),
            max(pos_std, 0.01),
            max(pos_std * np.sqrt(s), 0.01),
            max(ar_std, 0.01)
        ])
        
        self.R = np.diag(np.square(std))
    
    def _perform_single_update(self, measurement):
        """Perform single UKF update step."""
        # Transform sigma points to measurement space
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_meas = self._measurement_model(sigma_points)
        
        # Predicted measurement
        z_pred = np.dot(self.Wm, sigma_meas)
        diff_z = sigma_meas - z_pred
        Pzz = np.dot(diff_z.T * self.Wc, diff_z) + self.R
        
        # Cross-covariance
        diff_x = sigma_points - self.x
        Pxz = np.dot(diff_x.T * self.Wc, diff_z)
        
        # Kalman gain
        try:
            K = np.dot(Pxz, inv(Pzz))
        except np.linalg.LinAlgError:
            K = np.zeros((self.nx, self.nz))
        
        # Update state
        innovation = measurement - z_pred
        self.x += np.dot(K, innovation)
        self.P -= np.dot(K, np.dot(Pzz, K.T))
    
    def update(self, measurement=None, confidence=None, eta_pred=None, P_eta=None):
        """
        Update state with measurement and/or prediction from source tracker.
        
        Args:
            measurement: Local measurement in XYSR format
            confidence: Confidence of measurement
            eta_pred: Prediction from Source tracker (for Primary only)
            P_eta: Covariance from Source tracker (for Primary only)
        """
        # Update R based on measurement
        if measurement is not None:
            measurement = np.asarray(measurement).flatten()
            self._update_noise(measurement, confidence)
        
        if self.is_source:
            # Source tracker: Only update with real measurements
            if measurement is not None:
                self._perform_single_update(measurement)
        else:
            # Primary tracker: Can learn from Source
            if measurement is not None:
                # Has real measurement - use it
                self._perform_single_update(measurement)
            
            # Transfer Learning: Also incorporate Source prediction if available
            if eta_pred is not None and P_eta is not None:
                eta_pred = np.asarray(eta_pred).flatten()
                # Treat Source prediction as additional "measurement"
                # Use relaxed noise for virtual measurement
                R_virtual = self.R * 2.0  # Higher uncertainty for virtual
                self.R = R_virtual
                self._perform_single_update(eta_pred)
        
        # Ensure final covariance is PSD
        self.P = self._ensure_positive_definite(self.P)
        
        # Update history
        if measurement is not None:
            self.history_obs.append(measurement)
            self.last_measurement = measurement
            self.observed = True
        
        return self.x, self.P
    
    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            mean: State mean in XYSR format
            covariance: State covariance
            measurements: Array of measurements in XYSR format
            only_position: If True, only use position (x, y) for gating
            metric: 'maha' for Mahalanobis, 'gaussian' for Gaussian
            
        Returns:
            Array of distances
        """
        mean = np.asarray(mean).flatten()
        measurements = np.asarray(measurements)
        
        if len(measurements) == 0:
            return np.array([])
        
        # Ensure measurements is 2D
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)
        
        # Project state to measurement space
        if only_position:
            mean_meas = mean[:2]
            cov_meas = covariance[:2, :2]
            measurements = measurements[:, :2]
        else:
            mean_meas = mean[:self.nz]
            cov_meas = covariance[:self.nz, :self.nz]
            measurements = measurements[:, :self.nz]
        
        # Add measurement noise
        cov_meas = cov_meas + self.R[:len(mean_meas), :len(mean_meas)]
        
        # Ensure PSD
        cov_meas = self._ensure_positive_definite(cov_meas)
        
        # Compute distances
        try:
            cholesky_factor = np.linalg.cholesky(cov_meas)
            d = measurements - mean_meas
            z = np.linalg.solve(cholesky_factor, d.T).T
            squared_maha = np.sum(z * z, axis=1)
            
            if metric == 'gaussian':
                return squared_maha
            elif metric == 'maha':
                return squared_maha
            else:
                raise ValueError(f"Invalid metric: {metric}")
                
        except np.linalg.LinAlgError:
            # Fallback to Euclidean
            return np.sum((measurements - mean_meas) ** 2, axis=1)
    
    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        
        Returns:
            (projected_mean, projected_covariance, innovation_covariance)
        """
        mean = np.asarray(mean).flatten()
        
        # Project mean
        projected_mean = mean[:self.nz]
        
        # Project covariance
        projected_cov = covariance[:self.nz, :self.nz]
        
        # Innovation covariance (add measurement noise)
        innovation_cov = projected_cov + self.R
        innovation_cov = self._ensure_positive_definite(innovation_cov)
        
        return projected_mean, projected_cov, innovation_cov
