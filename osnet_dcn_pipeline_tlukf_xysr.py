import argparse
import os
import csv
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import StrongSort, StrongSortXYSR, StrongSortTLUKF
from datetime import datetime, timedelta
import numpy as np

class EndoStrongSort(StrongSortXYSR):
    def _crop_bbox(self, img, bbox):
        """Crop ảnh từ bbox dạng [x1, y1, x2, y2] với kiểm tra biên."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        return img[y1:y2, x1:x2].copy()
    """Enhanced StrongSort with virtual trajectory support for endoscopy videos."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_buffer = {}
        self.max_buffer_size = 10
        self.virtual_conf = 0.3
        self.appearance_thresh = 0.65
        self.max_gap = 8  # Maximum frames to interpolate for endoscopy
        self.frame_idx = 0
        self.last_features = {}
        self.lost_tracks = {}
        
    def _process_virtual_trajectories(self, detections, det_features, img):
        """Enhanced virtual trajectory processing with appearance matching."""
        # Process lost tracks
        lost_ids = list(self.lost_tracks.keys())
        for track_id in lost_ids:
            lost_info = self.lost_tracks[track_id]
            time_gap = self.frame_idx - lost_info['last_frame']
            if time_gap > self.max_gap:
                del self.lost_tracks[track_id]
                continue
            # Try to match with current detections
            if len(detections) > 0:
                best_match_idx = -1
                best_sim = self.appearance_thresh
                for i, feat in enumerate(det_features):
                    if lost_info['last_feature'] is not None:
                        sim = self._cosine_similarity(feat, lost_info['last_feature'])
                        if sim > best_sim:
                            best_sim = sim
                            best_match_idx = i
                if best_match_idx >= 0:
                    # Generate virtual trajectory
                    z1 = lost_info['last_pos']
                    z2 = detections[best_match_idx, :4]
                    virtual_boxes = self._interpolate_boxes(z1, z2, time_gap)
                    track = lost_info['track']
                    conf = self.virtual_conf
                    if best_sim > 0.8:
                        conf = min(0.7, self.virtual_conf * 1.5)
                    # Xác thực ngoại hình cho từng box ảo
                    for step, box in enumerate(virtual_boxes):
                        x_c, y_c, w, h = box
                        x1 = x_c - w/2
                        y1 = y_c - h/2
                        x2 = x_c + w/2
                        y2 = y_c + h/2
                        crop_frame_idx = lost_info['last_frame'] + step + 1
                        if crop_frame_idx in self.frame_buffer:
                            crop_img = self._crop_bbox(self.frame_buffer[crop_frame_idx], [x1, y1, x2, y2])
                            if crop_img.size > 0:
                                virtual_feat = self.model.get_features(np.array([[x1, y1, x2, y2]]), crop_img)[0]
                                sim_virtual = self._cosine_similarity(virtual_feat, lost_info['last_feature'])
                                if sim_virtual >= self.appearance_thresh:
                                    track.kf.update(box, confidence=conf)
                                    lost_info['virtual_boxes'].append((crop_frame_idx, box))
                    if best_sim > 0.85:
                        del self.lost_tracks[track_id]
            else:
                track = lost_info['track']
                predicted_box = track.kf.predict()[0]
                x_c, y_c, w, h = predicted_box
                x1 = x_c - w/2
                y1 = y_c - h/2
                x2 = x_c + w/2
                y2 = y_c + h/2
                crop_frame_idx = self.frame_idx
                if crop_frame_idx in self.frame_buffer:
                    crop_img = self._crop_bbox(self.frame_buffer[crop_frame_idx], [x1, y1, x2, y2])
                    if crop_img.size > 0:
                        virtual_feat = self.model.get_features(np.array([[x1, y1, x2, y2]]), crop_img)[0]
                        sim_virtual = self._cosine_similarity(virtual_feat, lost_info['last_feature'])
                        if sim_virtual >= self.appearance_thresh:
                            track.kf.update(predicted_box, confidence=self.virtual_conf * 0.7)
                            lost_info['virtual_boxes'].append((crop_frame_idx, predicted_box))
                if time_gap > self.max_gap:
                    del self.lost_tracks[track_id]
                    continue
            # # If no match found, predict position
            #     else:
            #         track = lost_info['track']
            #         predicted_box = track.kf.predict()[0]
            #         track.kf.update(predicted_box, confidence=self.virtual_conf * 0.7)
            #         lost_info['virtual_boxes'].append((self.frame_idx, predicted_box))
                    
            #         if time_gap > self.max_gap:
            #             del self.lost_tracks[track_id]
            #             continue
                    
                # Find best matching detection using appearance
                best_match = None
                best_sim = self.appearance_thresh
                
                for i, feat in enumerate(det_features):
                    sim = self._cosine_similarity(feat, lost_info['feature'])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = i
                        
                if best_match is not None:
                    # Generate virtual trajectory
                    z1 = lost_info['measurement']
                    z2 = detections[best_match, :4]  # Use matched detection
                    virtual_boxes = self._interpolate_boxes(z1, z2, time_gap)
                    
                    # Apply virtual boxes
                    track = lost_info['track']
                    conf = self.virtual_conf * (1 + best_sim) / 2  # Scale confidence by similarity
                    
                    for t, box in enumerate(virtual_boxes):
                        track.update(box, confidence=conf)
                        
                    # Remove from lost tracks
                    del self.lost_tracks[track_id]
        
        # Update lost tracks list with currently lost tracks
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update == 1:
                if hasattr(track, 'last_measurement') and track.last_measurement is not None:
                    self.lost_tracks[track.track_id] = {
                        'track': track,
                        'feature': track.features[-1] if track.features else None,
                        'measurement': track.last_measurement,
                        'last_frame': self.frame_idx - 1
                    }
                
                if 1 < time_gap <= self.max_gap:
                    # Find matching detection
                    matched_det = None
                    best_sim = self.appearance_thresh
                    
                    for det in detections:
                        if hasattr(det, 'feature') and track.features:
                            sim = self.metric.distance([det.feature], [track.features[-1]])[0]
                            if sim > best_sim:
                                matched_det = det
                                best_sim = sim
                    
                    if matched_det is not None:
                        # Generate and apply virtual trajectory
                        z1 = track.last_measurement
                        z2 = matched_det.to_xysr()
                        virtual_boxes = track.kf.interpolate_virtual_boxes(
                            z1, z2, 
                            track.last_seen_frame, self.frame_idx,
                            max_gap=self.max_gap
                        )
                        
                        for t, box in enumerate(virtual_boxes, start=track.last_seen_frame+1):
                            conf = self.virtual_conf
                            if best_sim > 0.8:  # High confidence in appearance match
                                conf = min(0.6, self.virtual_conf * 1.5)
                            track.kf.update(box, confidence=conf)
                            if t < self.frame_idx-1:
                                track.kf.predict()
    
    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two feature vectors."""
        if a is None or b is None:
            return 0.0
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    
    def _interpolate_boxes(self, z1, z2, steps):
        """Create interpolated boxes between two detections."""
        boxes = []
        for i in range(1, steps):
            t = i / steps
            box = z1 * (1 - t) + z2 * t
            boxes.append(box)
        return boxes

    def update(self, dets, img):
        """Update with always-on virtual trajectory support.
        
        Args:
            dets: numpy array of detections
            img: current frame image
        """
        self.frame_idx += 1
        # Update frame buffer
        if img is not None:
            self.frame_buffer[self.frame_idx] = img.copy()
            if len(self.frame_buffer) > self.max_buffer_size:
                del self.frame_buffer[min(self.frame_buffer.keys())]
        # Process detections
        dets_processed = self.preprocess_dets(dets)
        # Extract features for current detections
        if len(dets_processed) > 0:
            if hasattr(self.tracker, "model") and hasattr(self.tracker.model, "get_features"):
                det_features = self.tracker.model.get_features(img, dets_processed[:, :4])
            else:
                det_features = []
        else:
            det_features = []
        # Update virtual trajectories
        self._process_virtual_trajectories(dets_processed, det_features, img)
        # Luôn sinh box ảo cho các track bị miss
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 0:
                # Dự đoán vị trí ảo cho frame hiện tại
                predicted_box = track.kf.predict()[0]
                if not hasattr(track, 'virtual_boxes'):
                    track.virtual_boxes = []
                track.virtual_boxes.append((self.frame_idx, predicted_box))
        return super().update(dets, img)
        
    # Remove duplicate _cosine_similarity and _interpolate_boxes

    def preprocess_dets(self, dets):
        """
        Preprocess detections to standard format.
        """
        if dets is None:
            return []
        if isinstance(dets, list):
            dets = np.array(dets)
        if len(dets.shape) == 1:
            dets = np.expand_dims(dets, 0)
        if dets.shape[1] == 4:
            # Add confidence and class if not present
            conf = np.ones((dets.shape[0], 1))
            cls = np.zeros((dets.shape[0], 1))
            dets = np.hstack([dets, conf, cls])
        return dets

# Định nghĩa từ điển lớp cho các mô hình
MODEL_CLASSES_DICT = {
    "model_yolo/daday.pt": ['Viem da day', 'Ung thu da day'],
    "model_yolo/thucquan.pt": ['Viem thuc quan', 'Ung thu thuc quan'],
    "model_yolo/htt.pt": ['Loet HTT']
}

class Colors:
    """Quản lý bảng màu cho các lớp đối tượng."""
    def __init__(self, num_colors=80):
        self.num_colors = num_colors
        self.color_palette = self._generate_color_palette()

    def _generate_color_palette(self):
        """Tạo bảng màu HSV và chuyển sang BGR."""
        hsv_palette = np.zeros((self.num_colors, 1, 3), dtype=np.uint8)
        hsv_palette[:, 0, 0] = np.linspace(0, 180, self.num_colors, endpoint=False)
        hsv_palette[:, :, 1:] = 255
        return cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR).reshape(-1, 3)

    def __call__(self, class_id):
        """Trả về màu tương ứng với class_id."""
        return tuple(map(int, self.color_palette[class_id % self.num_colors]))

class ObjectDetection:
    """Lớp thực hiện phát hiện và theo dõi đối tượng trong video sử dụng YOLO và StrongSort hoặc TLUKF."""

    def __init__(self, model_weights, capture_path, output_dir, min_temporal_threshold=0, max_temporal_threshold=0,
                 iou_threshold=0.1, use_frame_id=False, tracker_type="xysr"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using Device: {self.device}")
        self.model = self._load_model(model_weights)
        self.classes = self.model.names  # Lấy danh sách lớp từ mô hình
        self.colors = Colors(len(self.classes))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.capture_path = Path(capture_path)
        self.output_dir = Path(output_dir)
        # Tạo folder riêng cho từng video
        self.video_folder = self.output_dir / self.capture_path.stem
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.mot_folder = self.output_dir / "mot"
        self.mot_folder.mkdir(parents=True, exist_ok=True)
        self.cap = self._load_capture()
        self.tracker_type = tracker_type
        self.tracker = self._initialize_tracker()
        self.min_temporal_threshold = min_temporal_threshold
        self.max_temporal_threshold = max_temporal_threshold
        self.iou_threshold = iou_threshold
        self.use_frame_id = use_frame_id
        # Virtual box confidence for missed frames
        self.virtual_conf = 0.1

    def _load_model(self, weights):
        """Tải và cấu hình mô hình YOLO."""
        model = YOLO(weights)
        model.fuse()
        return model
    
    def predict(self, frame):
        # TLUKF Strategy: Get ALL detections including low-confidence ones
        # - Source tracker: Will filter and use only conf ≥ 0.6 (high quality)
        # - Primary tracker: Will use ALL detections conf ≥ 0.3 (including low-conf)
        # - Transfer Learning: Primary learns from Source during gaps
        results = self.model(frame, stream=True, verbose=False, conf=0.45, line_width=1)
        return results

    def _initialize_tracker(self):
        """Khởi tạo tracker phù hợp: StrongSortXYSR hoặc TLUKF."""
        reid_weights = Path("osnet_dcn_x0_5_endocv.pt")
        if self.tracker_type == "tlukf":
            max_cos_dist,max_iou_dist,max_age,n_init,ema_alpha,mc_lambda = 0.9,	0.9, 500, 1, 0.95, 0.98,
            # Khởi tạo tracker TLUKF với thông số tối ưu cho endoscopy
            return StrongSortTLUKF(
                reid_weights=reid_weights,
                device=torch.device(self.device),
                half=False,
                max_cos_dist=max_cos_dist,
                max_iou_dist=max_iou_dist,
                max_age=max_age,
                n_init=n_init,
                ema_alpha=ema_alpha,
                mc_lambda=mc_lambda,
                per_class=False,
                # Appearance matching thresholds
                # max_cos_dist=0.8,  # CRITICAL: Tăng từ 0.2 → 0.4 để giảm ID switches
                #                    # Cosine distance: 0 = giống hệt, 1 = khác hoàn toàn
                #                    # 0.4 cho phép match với features tương tự nhưng không giống 100%
                # nn_budget=500,     # Giữ 500 features trong gallery
                # # Motion matching
                # max_iou_dist=0.5,  # IoU threshold cho motion matching
                # # Track lifecycle
                # max_age=500,       # Keep track alive 500 frames khi miss
                # n_init=3,          # CRITICAL: Giảm từ 3 → 1 để confirm track nhanh hơn
                #                    # Endoscopy: object xuất hiện ít frames → cần confirm nhanh
                # # Feature smoothing
                # ema_alpha=0.9,     # EMA weight cho feature smoothing
                # mc_lambda=0.5,   # Motion consistency weight
            )
        else:
            # Mặc định dùng StrongSortXYSR
            return StrongSortXYSR(
                reid_weights,
                torch.device(self.device),
                fp16=False,
                max_dist=0.95,
                max_iou_dist=0.95,
                max_age=300,
                half=False,
                # CRITICAL: Disable per-class tracking for cross-class matching
                per_class=False,
            )

    def _load_capture(self):
        """Tải video từ đường dẫn và cấu hình VideoWriter."""
        cap = cv2.VideoCapture(str(self.capture_path))
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {self.capture_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_name = f"tracking_{self.capture_path.stem}.mp4"
        self.video_name = video_name
        video_path = self.video_folder / video_name
        self.writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        return cap

    def _frame_idx_to_hms(self, frame_id):
        """Chuyển frame index thành timestamp hms."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        base = datetime.strptime('00:00:00', '%H:%M:%S')
        delta = timedelta(seconds=frame_id // fps)
        return (base + delta).strftime('%H:%M:%S')

    def _frame_idx_to_hmsf(self, frame_id):
        """Chuyển frame index thành timestamp hmsf."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        base = datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
        delta = timedelta(seconds=frame_id / fps)
        return (base + delta).strftime('%H:%M:%S.%f')

    def _write_seqinfo_ini(self, seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir):
        """Ghi thông tin sequence vào file seqinfo.ini."""
        seqinfo_path = self.video_folder / "seqinfo.ini"
        with open(seqinfo_path, "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={seq_name}\n")
            f.write(f"imDir={im_dir}\n")
            f.write(f"frameRate={frame_rate}\n")
            f.write(f"seqLength={seq_length}\n")
            f.write(f"imWidth={im_width}\n")
            f.write(f"imHeight={im_height}\n")
            f.write(f"imExt={im_ext}\n")

    def _calculate_iou(self, box1, box2):
        """Tính IoU giữa hai bounding box."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0
    
    def _apply_nms_to_tracks(self, tracks, iou_threshold=0.5):
        """
        CRITICAL: Keep ONLY ONE box per frame with highest priority.
        Priority: Real box (conf >= 0.35) > Virtual box (conf < 0.35)
        Within same category: Higher confidence > Lower confidence
        
        Args:
            tracks: numpy array of tracks [x1, y1, x2, y2, id, conf, cls, det_ind]
            iou_threshold: Not used - we always keep only 1 box
            
        Returns:
            Single track (best box) or empty array
        """
        if len(tracks) == 0:
            return tracks
        
        # Step 1: Separate real and virtual boxes
        real_boxes = []
        virtual_boxes = []
        
        for track in tracks:
            conf = track[5]
            if conf >= 0.35:
                real_boxes.append(track)
            else:
                virtual_boxes.append(track)
        
        # Step 2: Select ONLY ONE box with highest priority
        if len(real_boxes) > 0:
            # Priority 1: Real box with highest confidence
            real_boxes_sorted = sorted(real_boxes, key=lambda t: t[5], reverse=True)
            best_box = real_boxes_sorted[0]
        elif len(virtual_boxes) > 0:
            # Priority 2: Virtual box with highest confidence (if no real box)
            virtual_boxes_sorted = sorted(virtual_boxes, key=lambda t: t[5], reverse=True)
            best_box = virtual_boxes_sorted[0]
        else:
            return np.array([])
        
        # Return as 2D array for consistency
        return np.array([best_box])

    def _update_track_id(self, current_tracks, previous_tracks):
        """Cập nhật ID của các track dựa trên IoU."""
        updated_tracks = []
        for curr_track in current_tracks:
            min_distance = float('inf')
            matching_id = None
            for prev_track in previous_tracks:
                if curr_track[6] != prev_track[6]:  # Kiểm tra cùng lớp
                    continue
                iou = self._calculate_iou(curr_track[:4], prev_track[:4])
                if iou > self.iou_threshold:
                    time_diff = abs(curr_track[3] - prev_track[3]) if self.use_frame_id else abs(curr_track[1] - prev_track[1])
                    if time_diff < min_distance:
                        min_distance = time_diff
                        matching_id = prev_track[4]
            curr_track[4] = matching_id if matching_id is not None else curr_track[4]
            updated_tracks.append(curr_track)
        return updated_tracks

    def _draw_tracks(self, frame, tracks, txt_file):
        """Vẽ các track lên frame và ghi vào file txt, bao gồm cả hộp giới hạn ảo nếu có."""
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp_hms = self._frame_idx_to_hms(frame_id)
        timestamp_hmsf = self._frame_idx_to_hmsf(frame_id)
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        # Process all tracks and separate real vs virtual based on confidence
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            id_ = int(track[4])
            conf = round(track[5], 2)
            class_id = int(track[6])
            class_name = self.classes[class_id]
            
            # All tracks use same color and thickness
            color = self.colors(class_id)
            thickness = 5
            font_scale = 1.5
            font_thickness = 3
            
            # TLUKF: Distinguish by confidence and update label prefix
            # - High conf (≥0.6): Strong detection (Source + Primary updated)
            # - Low conf (0.3-0.6): Weak detection (only Primary updated) - prefix (L)
            # - Very low conf (<0.35): Virtual/predicted box (TLUKF prediction) - prefix (V)
            
            if conf >= 0.6:
                # Strong detection - both trackers updated
                label = f'{class_name}, ID: {id_}, conf: {conf}'
                notes = "Tracking"
            elif conf >= 0.45:
                # Weak detection - only Primary updated (TLUKF advantage)
                label = f'(L) {class_name}, ID: {id_}, conf: {conf}'
                notes = "Tracking Low Confidence"  # Still a real detection, just low confidence
            else:
                # Virtual box - TLUKF prediction (Transfer Learning active)
                label = f'(V) {class_name}, ID: {id_}, conf: {conf}'
                notes = "Virtual"
            
            # Draw rectangle with uniform style
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with uniform style
            (w, h), _ = cv2.getTextSize(label, self.font, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 + h + 15), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 + h + 10), self.font, font_scale, (255, 255, 255), font_thickness)
            
            # Log to file
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            txt_file.write(f"{timestamp_hms},{timestamp_hmsf},{frame_id},{frame_rate},{class_name},{id_},{id_},{notes},"
                           f"{frame.shape[0]},{frame.shape[1]},{frame.shape[0]},{frame.shape[1]},{x1},{y1},{x2},{y2},"
                           f"{center_x},{center_y}\n")

        return frame

    def _txt_to_csv(self, txt_file, csv_file):
        """Chuyển đổi file txt sang csv."""
        with open(txt_file, 'r') as txt, open(csv_file, 'w', newline='') as csv_f:
            reader = csv.reader(txt)
            writer = csv.writer(csv_f)
            header = next(reader)
            writer.writerow(header)
            writer.writerows(reader)

    def _convert_to_mot(self, txt_file, mot_file):
        """Chuyển đổi file txt sang định dạng MOT."""
        with open(txt_file, 'r') as txt, open(mot_file, 'w', newline='') as mot:
            reader = csv.reader(txt)
            next(reader)  # Bỏ qua header
            for row in reader:
                frame_id = row[2]
                track_id = row[6]
                x1, y1, x2, y2 = map(float, row[12:16])
                conf = float(row[5])
                bb_width = x2 - x1
                bb_height = y2 - y1
                mot.write(f"{frame_id},{track_id},{x1},{y1},{bb_width},{bb_height},{conf},-1,-1,-1\n")

    def __call__(self):
        """Thực thi quá trình phát hiện và theo dõi."""
        tracker = self.tracker
        seq_name = "StrongSort"
        im_dir = "img1"
        seq_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        im_ext = ".jpg"

        self._write_seqinfo_ini(seq_name, seq_length, frame_rate, im_width, im_height, im_ext, im_dir)
        txt_file_path = self.video_folder / "tracking_result.txt"
        mot_file_path = self.mot_folder / f"{self.capture_path.stem}.txt"
        csv_file_path = self.video_folder / f"tracking_{self.capture_path.stem}.csv"

        with open(txt_file_path, "w") as txt_file:
            txt_file.write("timestamp_hms,timestamp_hmsf,frame_idx,fps,object_cls,object_idx,object_id,notes,"\
                           "frame_height,frame_width,scale_height,scale_width,x1,y1,x2,y2,center_x,center_y\n")
            previous_tracks = []

            while True:
                start_time = perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    break

                cv2.rectangle(frame, (0, 30), (220, 80), (255, 255, 255), -1)
                detections = self.predict(frame)

                for dets in detections:
                    det_boxes = dets.boxes.data.to("cpu").numpy()
                    
                    # TLUKF: Pass ALL detections to tracker
                    # - Tracker will internally handle Source (conf ≥ 0.6) vs Primary (all conf ≥ 0.3)
                    # - No need to split here - let TLUKF's dual-tracker architecture handle it
                    if det_boxes.size > 0:
                        tracks = tracker.update(det_boxes, frame)
                    else:
                        tracks = tracker.update(np.empty((0, 6), dtype=np.float32), frame)
                    
                    # CRITICAL FIX: Apply NMS to keep ONLY ONE box per frame
                    if len(tracks.shape) == 2 and tracks.shape[1] == 8 and tracks.size > 0:
                        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        
                        # Apply NMS - returns ONLY ONE box (highest priority)
                        tracks = self._apply_nms_to_tracks(tracks, iou_threshold=0.001)
                        
                        # Update track IDs and draw
                        if len(previous_tracks) > 0:
                            tracks = self._update_track_id(tracks, previous_tracks)
                        frame = self._draw_tracks(frame, tracks, txt_file)
                        previous_tracks = tracks
                    else:
                        # No real tracks this frame; still draw to allow virtual logging
                        frame = self._draw_tracks(frame, np.empty((0, 8), dtype=np.float32), txt_file)

                end_time = perf_counter()
                fps = 1 / (end_time - start_time)
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), self.font, 1.5, (0, 255, 0), 5)
                self.writer.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        self._txt_to_csv(str(txt_file_path), str(csv_file_path))
        self._convert_to_mot(str(txt_file_path), str(mot_file_path))

        print(f"Finished: {self.video_name} -> {self.video_folder}")

def main():
    """Chức năng chính để xử lý video với các tham số từ argparse."""
    parser = argparse.ArgumentParser(description="Object Detection and Tracking using YOLO and StrongSort/TLUKF")
    parser.add_argument("--video_dir", type=str, default="data/video_test", help="Thư mục chứa video đầu vào")
    parser.add_argument("--model_dir", type=str, default="model_yolo", help="Thư mục chứa mô hình YOLO")
    parser.add_argument("--output_dir", type=str, default="content3112/runs_tlukf_xysr_vbox_short", help="Thư mục đầu ra cho kết quả")
    parser.add_argument("--min_temporal_threshold", type=float, default=0, help="Ngưỡng thời gian tối thiểu")
    parser.add_argument("--max_temporal_threshold", type=float, default=0, help="Ngưỡng thời gian tối đa")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="Ngưỡng IoU cho cập nhật ID")
    parser.add_argument("--use_frame_id", action="store_true", help="Sử dụng frame ID để tính thời gian")
    parser.add_argument("--tracker_type", type=str, default="tlukf", choices=["xysr", "tlukf"], help="Chọn loại tracker: xysr hoặc tlukf")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Tạo thư mục đầu ra nếu chưa tồn tại
    video_extensions = (".mp4", ".avi", ".mov")

    for video_path in video_dir.rglob("*"):
        if video_path.suffix.lower() in video_extensions:
            model_weights = (
                model_dir / "thucquan.pt" if "UTTQ" in video_path.parts
                else model_dir / "daday.pt" if "UTDD" in video_path.parts
                else model_dir / "htt.pt"
            )
            print(f"Processing video: {video_path} with model: {model_weights} | Tracker: {args.tracker_type}")
            detector = ObjectDetection(
                model_weights=str(model_weights),
                capture_path=video_path,
                output_dir=output_dir,
                min_temporal_threshold=args.min_temporal_threshold,
                max_temporal_threshold=args.max_temporal_threshold,
                iou_threshold=args.iou_threshold,
                use_frame_id=args.use_frame_id,
                tracker_type=args.tracker_type
            )
            start_video_time = perf_counter()
            detector()
            end_video_time = perf_counter()
            print(f"Time taken for {video_path.name}: {end_video_time - start_video_time:.2f} seconds")

if __name__ == "__main__":
    main()