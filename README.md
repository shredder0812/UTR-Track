# UTR-Track

## Overview

**UTR-Track** (Universal Tracking with Transfer Learning) is an advanced object tracking system optimized for medical videos (endoscopy). The project combines YOLO for object detection with advanced trackers (StrongSort, UTRTrack) to track pathologies in endoscopic videos.

### Key Features

- **Multi-Pathology Detection**: Supports detection and tracking of various pathologies:
  - Esophagitis, Esophageal Cancer
  - Gastritis, Gastric Cancer
  - Duodenal Ulcer

- **Intelligent Tracking**:
  - **UTRTrack (TLUKF)**: Advanced tracker with virtual trajectory prediction capability when detection is lost
  - **StrongSort**: Stable traditional tracker with multiple variants (XYSR)
  - Cross-class tracking: Track objects across different classes
  
- **Virtual Trajectory Interpolation**: Automatically predict object positions when occluded or temporarily lost

- **Optimized for Endoscopy Videos**:
  - Handle motion blur and low contrast
  - Adaptive confidence scoring
  - Appearance-based track recovery

###  Project Structure

```
UTR-Track/
├── utrtrack_pipeline.py           # Main file to run tracking
├── osnet_dcn_x0_5_endocv.pt      # ReID weights for appearance matching
├── model_yolo/                    # Directory containing YOLO models
│   ├── daday.pt                  # Model for stomach
│   ├── thucquan.pt               # Model for esophagus
│   └── htt.pt                    # Model for duodenum
├── data/                          # Data directory
│   └── video_test/               # Input videos
└── boxmot/                        # Tracking library
    └── boxmot/
        └── trackers/
            ├── utrtrack/         # UTRTrack (TLUKF) tracker
            ├── strongsort/       # StrongSort tracker
            └── ...               # Other trackers
```

## Installation

### System Requirements

- Python >= 3.8
- CUDA (recommended for GPU) or CPU
- pip / uv package manager

### Installation Steps

1. **Clone repository**:
```bash
git clone https://github.com/shredder/UTR-Track.git
cd UTR-Track
```

2. **Install dependencies**:

**Method 1: Using pip (recommended)**
```bash
# Install basic packages
pip install -r requirements.txt

# Or manually install main packages:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU
# pip install torch torchvision  # CPU

pip install ultralytics opencv-python numpy pandas
```

**Method 2: Install boxmot from source**
```bash
cd boxmot
pip install -e .
cd ..
```

3. **Download ReID weights** (if not available):
```bash
# File osnet_dcn_x0_5_endocv.pt should be placed in the root directory
# If not available, download from link or train ReID model
```

4. **Prepare YOLO models**:
```bash
# Place YOLO models in model_yolo/ directory
mkdir -p model_yolo
# Copy or train models: daday.pt, thucquan.pt, htt.pt
```

## Usage

### 1. Basic Tracking

Run tracking with default configuration (using UTRTrack - TLUKF):

```bash
python utrtrack_pipeline.py \
    --video_dir data/video_test \
    --model_dir model_yolo \
    --output_dir content/runs_tlukf \
    --tracker_type tlukf
```

### 2. Tracking with StrongSort

```bash
python utrtrack_pipeline.py \
    --video_dir data/video_test \
    --model_dir model_yolo \
    --output_dir content/runs_strongsort \
    --tracker_type xysr
```

### 3. Custom Parameters

```bash
python utrtrack_pipeline.py \
    --video_dir data/video_test \
    --model_dir model_yolo \
    --output_dir content/custom_run \
    --tracker_type tlukf \
    --iou_threshold 0.3 \
    --min_temporal_threshold 5 \
    --max_temporal_threshold 30
```

### 4. Parameter Details

| Parameter | Default | Description |
|---------|----------|-------|
| `--video_dir` | `data/video_test` | Directory containing input videos |
| `--model_dir` | `model_yolo` | Directory containing YOLO models |
| `--output_dir` | `content/runs` | Output directory for results |
| `--tracker_type` | `tlukf` | Tracker type: `tlukf` (UTRTrack) or `xysr` (StrongSort) |
| `--iou_threshold` | `0.1` | IoU threshold for track matching |
| `--min_temporal_threshold` | `0` | Minimum temporal threshold (frames) |
| `--max_temporal_threshold` | `0` | Maximum temporal threshold (frames) |
| `--use_frame_id` | `False` | Use frame ID instead of timestamp |

## Output

After running, the system generates the following files:

```
output_dir/
├── [video_name]/                      # Directory for each video
│   ├── tracking_[video_name].mp4     # Video with drawn tracking
│   ├── tracking_result.txt            # Tracking results (custom format)
│   ├── tracking_[video_name].csv     # Tracking results (CSV)
│   └── seqinfo.ini                    # Sequence metadata
└── mot/                               # MOT Challenge format
    └── [video_name].txt              # Tracking results (MOT format)
```

### CSV Output Format:

```csv
timestamp_hms,timestamp_hmsf,frame_idx,fps,object_cls,object_idx,object_id,notes,
frame_height,frame_width,scale_height,scale_width,x1,y1,x2,y2,center_x,center_y
```

### Tracking Status Types:

- **Tracking**: Detection with high confidence (>= 0.6)
- **Tracking Low Confidence**: Detection with low confidence (0.45-0.6)
- **(L) prefix**: Low confidence detection (UTRTrack only)
- **(V) prefix**: Virtual/predicted box (trajectory interpolation - UTRTrack only)

## 🔧 Advanced Configuration

### Adjust Tracker Parameters

Edit in file [utrtrack_pipeline.py](utrtrack_pipeline.py):

**For UTRTrack (TLUKF)**:
```python
return UTRTrack(
    reid_weights=reid_weights,
    device=torch.device(self.device),
    half=False,
    max_cos_dist=0.9,      # Cosine distance threshold
    max_iou_dist=0.9,      # IoU distance threshold
    max_age=500,           # Max frames to keep track alive
    n_init=1,              # Min detections to confirm track
    ema_alpha=0.95,        # EMA smoothing factor
    mc_lambda=0.98,        # Motion consistency weight
    per_class=False        # Cross-class tracking
)
```

**For StrongSort (XYSR)**:
```python
return StrongSortXYSR(
    reid_weights,
    torch.device(self.device),
    fp16=False,
    max_dist=0.95,         # Max cosine distance
    max_iou_dist=0.95,     # Max IoU distance
    max_age=300,           # Max age for track
    half=False,
    per_class=False
)
```


## References

- [BoxMOT - Multi-Object Tracking](https://github.com/mikel-brostrom/boxmot)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [StrongSort Paper](https://arxiv.org/abs/2202.13514)


## Contact

For questions or issues, please contact: [tung.ct242509m@sis.hust.edu.vn]