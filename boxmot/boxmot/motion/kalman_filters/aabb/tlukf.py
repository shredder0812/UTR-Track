import numpy as np
import scipy.linalg

# Bảng giá trị Chi-squared cho cổng Mahalanobis (gate) ở mức tin cậy 95%
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

class TLUKFTracker:
    """
    Triển khai Unscented Kalman Filter (UKF) cho việc theo dõi đối tượng.
    - Trạng thái: [x, y, a, h, vx, vy, va, vh]
      (tâm x, tâm y, tỷ lệ khung hình, chiều cao, và các vận tốc tương ứng).
    - Có khả năng hoạt động như một tracker 'source' hoặc 'primary',
      với tracker 'primary' có thể hợp nhất thông tin từ một 'source'.
    """
    def __init__(self, is_source=False):
        # --- Cấu hình tham số ---
        self.nx = 8  # Kích thước vector trạng thái
        self.nz = 4  # Kích thước vector đo lường [x, y, a, h]
        
        # Tham số UKF
        self.alpha = 1e-1
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.nx + self.kappa) - self.nx
        
        # Trọng số nhiễu (dùng để điều chỉnh R và P ban đầu)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.dt = 1.0

        # Cờ xác định vai trò của tracker
        self.is_source = is_source
        
        # Hằng số cho việc ổn định số học
        self._PSD_MIN_EIG = 1e-3 # Giá trị riêng tối thiểu để đảm bảo xác định dương
        self._CHOLESKY_REG_FACTOR = 1e-4 # Hệ số điều chuẩn khi Cholesky thất bại

        # --- Khởi tạo ma trận nhiễu ---
        # Nhiễu quá trình (Process Noise Covariance Q)
        # Theo TL-UKF paper: vận tốc kích thước (va, vh) cần nhiễu CỰC THẤP
        # để tránh box ảo thay đổi kích thước phi lý
        self.Q = np.diag([
            0.5, 0.5,        # Vị trí (x, y) - có thể thay đổi
            1e-6, 1e-6,      # Aspect ratio & height (a, h) - GẦN KHÔNG ĐỔI
            1.0, 1.0,        # Vận tốc vị trí (vx, vy) - cho phép di chuyển
            1e-8, 1e-8       # Vận tốc kích thước (va, vh) - CỰC THẤP để box ổn định
        ]) * self.dt

        # Nhiễu đo lường (Measurement Noise Covariance R)
        # Sẽ được cập nhật động trong hàm update
        std = [
            self._std_weight_position, self._std_weight_position, 
            0.1, self._std_weight_position
        ]
        self.R = np.diag(np.square(std))

        # Khởi tạo trạng thái
        self.x = np.zeros(self.nx)
        self.P = np.eye(self.nx)
        self.Wm, self.Wc = self._compute_weights()

    def _compute_weights(self):
        """Tính toán các trọng số cho các điểm sigma."""
        Wm = np.full(2 * self.nx + 1, 0.5 / (self.nx + self.lambda_))
        Wc = Wm.copy()
        Wm[0] = self.lambda_ / (self.nx + self.lambda_)
        Wc[0] = Wm[0] + (1 - self.alpha**2 + self.beta)
        return Wm, Wc

    def _ensure_positive_definite(self, matrix):
        """Đảm bảo ma trận hiệp phương sai là đối xứng và xác định dương."""
        # Ensure symmetry
        matrix = 0.5 * (matrix + matrix.T)
        
        # Check for NaN or Inf
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            # Return identity matrix scaled appropriately if matrix is invalid
            return np.eye(matrix.shape[0]) * self._PSD_MIN_EIG * 100
        
        # Add small regularization to diagonal for numerical stability
        matrix += np.eye(matrix.shape[0]) * self._CHOLESKY_REG_FACTOR
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            # Clip eigenvalues to minimum threshold
            eigenvalues = np.maximum(eigenvalues, self._PSD_MIN_EIG)
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            # If eigenvalue decomposition fails, return regularized identity
            return np.eye(matrix.shape[0]) * self._PSD_MIN_EIG * 100

    def _generate_sigma_points(self, x, P):
        """Tạo ra các điểm sigma từ trạng thái (mean) và hiệp phương sai (covariance)."""
        # Validate state vector
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # Reset to reasonable default if state is invalid
            x = np.zeros(self.nx)
            x[2] = 1.0  # aspect ratio = 1
            x[3] = 100.0  # height = 100
        
        P_stable = self._ensure_positive_definite(P)
        
        try:
            # Sử dụng phân rã Cholesky để tìm căn bậc hai của ma trận
            sqrt_P = scipy.linalg.cholesky(P_stable, lower=True)
        except (np.linalg.LinAlgError, ValueError):
            # Nếu thất bại, thêm regularization mạnh hơn và thử lại
            P_stable += np.eye(self.nx) * self._CHOLESKY_REG_FACTOR * 10
            try:
                sqrt_P = scipy.linalg.cholesky(P_stable, lower=True)
            except (np.linalg.LinAlgError, ValueError):
                # Last resort: use scaled identity
                sqrt_P = np.eye(self.nx) * np.sqrt(self._PSD_MIN_EIG * 100)
        
        sigma_points = np.zeros((2 * self.nx + 1, self.nx))
        sigma_points[0] = x
        scale = np.sqrt(self.nx + self.lambda_)
        for i in range(self.nx):
            sigma_points[i + 1] = x + scale * sqrt_P[:, i]
            sigma_points[self.nx + i + 1] = x - scale * sqrt_P[:, i]
        
        return sigma_points

    def _motion_model(self, sigma_points):
        """Hàm chuyển đổi trạng thái (mô hình chuyển động vận tốc không đổi)."""
        transitioned = np.copy(sigma_points)
        dt = self.dt
        transitioned[:, :4] += dt * sigma_points[:, 4:] # p' = p + v*dt
        # v' = v (vận tốc không đổi)
        return transitioned

    def _measurement_model(self, sigma_points):
        """Hàm đo lường (trích xuất [x, y, a, h] từ trạng thái)."""
        return sigma_points[:, :self.nz]

    def initiate(self, measurement):
        """Khởi tạo trạng thái của tracker từ một phép đo ban đầu."""
        measurement = np.asarray(measurement)
        mean_pos = measurement[:self.nz]
        mean_vel = np.zeros(self.nz)
        mean = np.r_[mean_pos, mean_vel]

        height = measurement[3]
        std = [
            2 * self._std_weight_position * height,
            2 * self._std_weight_position * height,
            1e-2,
            2 * self._std_weight_position * height,
            10 * self._std_weight_velocity * height,
            10 * self._std_weight_velocity * height,
            1e-5,
            10 * self._std_weight_velocity * height
        ]
        covariance = np.diag(np.square(std))
        
        self.x = mean
        self.P = covariance
        return self.x, self.P

    def predict(self):
        """Dự đoán trạng thái và hiệp phương sai ở bước thời gian tiếp theo."""
        # Tạo và lan truyền các điểm sigma qua mô hình chuyển động
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_pred = self._motion_model(sigma_points)

        # Tính toán lại mean và covariance từ các điểm sigma đã lan truyền
        mean_pred = np.dot(self.Wm, sigma_pred)
        diff = sigma_pred - mean_pred
        cov_pred = np.dot(diff.T * self.Wc, diff)
        cov_pred += self.Q  # Thêm nhiễu quá trình

        self.x = mean_pred
        self.P = cov_pred

    def _update_noise(self, measurement, confidence):
        """Cập nhật ma trận nhiễu đo lường R một cách động."""
        height = float(measurement[3])
        conf_value = float(confidence) if confidence is not None else 0.0
        
        # Độ bất định tăng khi confidence giảm
        uncertainty_factor = 1.0 - np.clip(conf_value, 0.0, 1.0)
        
        pos_std = self._std_weight_position * height * (1.0 + uncertainty_factor)
        ar_std = 0.1 * (1.0 + uncertainty_factor)
        
        # Đảm bảo độ lệch chuẩn có giá trị tối thiểu
        std = np.array([
            max(pos_std, 0.01), # x
            max(pos_std, 0.01), # y
            max(ar_std, 0.01),  # aspect ratio
            max(pos_std, 0.01)  # height
        ])
        
        self.R = np.diag(np.square(std))

    def _perform_single_update(self, measurement):
        """Thực hiện một bước cập nhật UKF với một phép đo duy nhất."""
        # Biến đổi các điểm sigma sang không gian đo lường
        sigma_points = self._generate_sigma_points(self.x, self.P)
        sigma_meas = self._measurement_model(sigma_points)

        # Dự đoán phép đo và hiệp phương sai
        z_pred = np.dot(self.Wm, sigma_meas)
        diff_z = sigma_meas - z_pred
        Pzz = np.dot(diff_z.T * self.Wc, diff_z) + self.R

        # Hiệp phương sai chéo
        diff_x = sigma_points - self.x
        Pxz = np.dot(diff_x.T * self.Wc, diff_z)

        # Tính Kalman Gain và cập nhật trạng thái
        try:
            K = np.linalg.solve(Pzz.T, Pxz.T).T
        except np.linalg.LinAlgError:
            # Nếu ma trận Pzz suy biến, bỏ qua bước cập nhật
            return

        innovation = measurement - z_pred
        self.x += np.dot(K, innovation)
        self.P -= np.dot(K, np.dot(Pzz, K.T))

    def update(self, measurement=None, confidence=None, eta_pred=None, P_eta=None):
        """
        Cập nhật trạng thái bằng phép đo và/hoặc dự đoán từ tracker khác (eta).
        Sử dụng phương pháp cập nhật tuần tự.
        """
        # Cập nhật R dựa trên phép đo cục bộ nếu có
        if measurement is not None:
            measurement = np.asarray(measurement)[:self.nz]
            self._update_noise(measurement, confidence)

        if self.is_source:
            # Tracker 'source' chỉ cập nhật với phép đo của chính nó
            if measurement is not None:
                self._perform_single_update(measurement)
        else:
            # Tracker 'primary' thực hiện cập nhật tuần tự
            # 1. Cập nhật với dự đoán từ 'source' (eta_pred) nếu có
            if eta_pred is not None and P_eta is not None:
                # Validate eta_pred and P_eta
                eta_pred_arr = np.asarray(eta_pred)[:self.nz]
                if not (np.any(np.isnan(eta_pred_arr)) or np.any(np.isinf(eta_pred_arr))):
                    P_eta_sub = P_eta[:self.nz, :self.nz]
                    if not (np.any(np.isnan(P_eta_sub)) or np.any(np.isinf(P_eta_sub))):
                        # Tạm thời coi P_eta là nhiễu đo lường từ 'source'
                        original_R = self.R.copy()
                        self.R = self._ensure_positive_definite(P_eta_sub)
                        self._perform_single_update(eta_pred_arr)
                        self.R = original_R # Khôi phục R ban đầu

            # 2. Cập nhật với phép đo cục bộ (measurement) nếu có
            if measurement is not None:
                self._perform_single_update(measurement)
        
        # Đảm bảo hiệp phương sai cuối cùng vẫn xác định dương
        self.P = self._ensure_positive_definite(self.P)
        return self.x, self.P

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        Tính khoảng cách Mahalanobis giữa trạng thái dự đoán và các phép đo.
        Được sử dụng để loại bỏ các phép đo ngoại lai (outliers).
        
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in format (x, y, a, h).
        only_position : Optional[bool]
            If True, distance computation is done with respect to position only.
        metric : str
            Distance metric ('maha' for Mahalanobis, others not implemented).
            
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and measurement[i].
        """
        if len(measurements) == 0:
            return np.array([])
        
        measurements = np.asarray(measurements)
        
        # Use provided mean and covariance or fall back to internal state
        if mean is None:
            mean = self.x
        if covariance is None:
            covariance = self.P
        
        # Dự đoán phép đo và hiệp phương sai của nó từ state đã cho
        sigma_points = self._generate_sigma_points(mean, covariance)
        sigma_meas = self._measurement_model(sigma_points)
        z_pred = np.dot(self.Wm, sigma_meas)
        diff_z = sigma_meas - z_pred
        S = np.dot(diff_z.T * self.Wc, diff_z) + self.R

        if only_position:
            z_pred = z_pred[:2]
            measurements = measurements[:, :2]
            S = S[:2, :2]
            
        S_stable = self._ensure_positive_definite(S)

        try:
            chol_S = scipy.linalg.cholesky(S_stable, lower=True)
            inv_chol_S = scipy.linalg.solve_triangular(chol_S, np.eye(S.shape[0]), lower=True)
            inv_S = inv_chol_S.T @ inv_chol_S
        except (np.linalg.LinAlgError, ValueError):
            # Nếu không thể phân rã, trả về khoảng cách vô cùng
            return np.full(len(measurements), np.inf)

        # Tính khoảng cách Mahalanobis cho tất cả phép đo cùng lúc
        innovations = measurements - z_pred
        # d^2 = v^T * S^-1 * v
        distances = np.sum((innovations @ inv_S) * innovations, axis=1)
        
        return distances