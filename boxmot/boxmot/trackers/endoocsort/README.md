# EndoOcSort Tracker

## Overview

**EndoOcSort** là tracker mới kết hợp ưu điểm của **OcSort** và **TLUKF (Transfer Learning UKF)** được thiết kế đặc biệt cho endoscopy video tracking.

## Architecture

```
EndoOcSort
├── Motion Model: XYSR-TLUKF (Unscented Kalman Filter)
├── Association: OcSort-style (Velocity + Observation History)
├── ReID: OsNet DCN (Deep Convolutional Network)
└── Virtual Trajectory: Transfer Learning Support
```

## Key Features

### 1. **XYSR-TLUKF Motion Model**
- State space: `[x, y, s, r, vx, vy, vs, vr]`
  - `x, y`: Center position
  - `s`: Scale (area = w × h)
  - `r`: Aspect ratio (w / h)
  - `vx, vy, vs, vr`: Velocities
- Unscented Kalman Filter (UKF) cho non-linear motion
- Transfer Learning: Primary tracker học từ Source tracker

### 2. **OcSort Association**
- **Velocity-based matching**: Sử dụng vận tốc để cải thiện association
- **Observation history**: Tracking lịch sử quan sát để re-association
- **BYTE-style matching**: Second-stage matching với low-confidence detections
- **Adaptive inertia**: Trọng số động cho velocity component

### 3. **Transfer Learning Strategy**
- **Source Tracker**: Chỉ sử dụng high-confidence detections (conf ≥ 0.6)
- **Primary Tracker**: Sử dụng tất cả detections (conf ≥ 0.3)
- **Knowledge Transfer**: Primary học từ Source predictions khi bị miss
- **Virtual Boxes**: Duy trì tracking trong occlusions

### 4. **Dual-Stage Matching**
1. **First stage**: High-confidence detections với appearance + IOU
2. **Second stage**: Low-confidence detections với BYTE association
3. **Third stage**: Observation history re-association

## File Structure

```
boxmot/boxmot/trackers/endoocsort/
├── __init__.py                    # Package initialization
├── endoocsort.py                  # Main tracker wrapper
└── sort/
    ├── __init__.py
    ├── track.py                   # TrackEndoOcSort class
    ├── tracker.py                 # TrackerEndoOcSort core logic
    ├── detection.py               # Detection class
    ├── iou_matching.py            # IOU matching utilities
    └── linear_assignment.py       # Hungarian algorithm

boxmot/boxmot/motion/kalman_filters/aabb/
└── xysr_tlukf.py                  # XYSR-TLUKF motion model (NEW)
```

## Usage

### Basic Usage

```python
from boxmot.trackers.endoocsort import EndoOcSort
import torch

tracker = EndoOcSort(
    reid_weights="osnet_dcn_x0_5_endocv.pt",
    device=torch.device("cuda:0"),
    half=False,
    det_thresh=0.45,
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    max_iou_dist=0.7,
    max_cos_dist=0.3,
    n_init=3,
    ema_alpha=0.9,
    mc_lambda=0.995,
    delta_t=3,
    inertia=0.2,
    use_byte=True,
    min_conf=0.1
)

# Update tracker
tracks = tracker.update(detections, frame)
```

### Pipeline Usage

```bash
# Run with EndoOcSort tracker
python osnet_dcn_pipeline_ocsort_vta.py \
    --video_dir data/video_test \
    --model_dir model_yolo \
    --output_dir content3112/runs_endoocsort_tlukf
```

## Parameters

### BaseTracker Parameters
- `det_thresh`: Detection threshold (default: 0.45)
- `max_age`: Maximum age before deletion (default: 30)
- `max_obs`: Maximum observations to store (default: 50)
- `min_hits`: Minimum hits before confirmation (default: 3)
- `iou_threshold`: IOU threshold for association (default: 0.3)

### EndoOcSort-Specific Parameters
- `max_cos_dist`: Maximum cosine distance for ReID (default: 0.3)
- `max_iou_dist`: Maximum IOU distance (default: 0.7)
- `n_init`: Frames to confirm track (default: 3)
- `nn_budget`: Feature library size (default: 100)
- `ema_alpha`: EMA smoothing factor (default: 0.9)
- `mc_lambda`: Motion consistency weight (default: 0.995)
- `delta_t`: Observation history window (default: 3)
- `inertia`: Velocity weight in association (default: 0.2)
- `use_byte`: Use BYTE association (default: True)
- `min_conf`: Min confidence for second-stage (default: 0.1)

## Advantages vs OcSort

1. **Better Motion Model**: UKF vs linear KF → Better handling of non-linear motion
2. **Transfer Learning**: Can learn from high-quality tracks → Better recovery after occlusion
3. **Virtual Trajectory**: Maintains tracking during gaps → Reduced ID switches
4. **XYSR State**: Better scale/ratio estimation → More stable boxes
5. **Appearance Features**: ReID support → Better long-term re-identification

## Advantages vs StrongSort-TLUKF

1. **Velocity Association**: Better short-term matching → Faster response
2. **Observation History**: Better re-association → Recovered lost tracks
3. **BYTE Matching**: Better use of low-conf detections → Higher recall
4. **Simpler Architecture**: Easier to tune → Better for endoscopy

## Performance Tuning

### For High Occlusion
```python
tracker = EndoOcSort(
    max_age=50,          # Longer track lifetime
    delta_t=5,           # Larger observation window
    inertia=0.3,         # Higher velocity weight
    use_byte=True,       # Enable BYTE matching
)
```

### For Fast Motion
```python
tracker = EndoOcSort(
    delta_t=2,           # Shorter observation window
    inertia=0.4,         # Higher velocity weight
    mc_lambda=0.98,      # Stronger motion consistency
)
```

### For Crowded Scenes
```python
tracker = EndoOcSort(
    max_iou_dist=0.6,    # Stricter IOU matching
    max_cos_dist=0.2,    # Stricter appearance matching
    n_init=5,            # More hits before confirmation
)
```

## Output Format

Same as StrongSort:
```
[x1, y1, x2, y2, track_id, confidence, class_id, detection_index]
```

## Notes

- **XYSR-TLUKF** là motion model mới được tạo riêng cho EndoOcSort
- File `xysr_tlukf.py` kết hợp UKF với Transfer Learning trong XYSR state space
- Tất cả logic tracking được tổ chức trong folder `endoocsort/` độc lập
- Compatible với boxmot framework

## Author

Created for EndoCV 2025 Challenge - Endoscopy Video Tracking
