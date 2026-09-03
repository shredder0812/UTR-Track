# UTR-Track

## Overview

**UTR-Track** (Universal Tracking with Transfer Learning) is an advanced object tracking system optimized for medical videos (endoscopy). The project combines YOLO for object detection with advanced trackers (StrongSort, UTRTrack) to track pathologies in endoscopic videos.

### Tracking Results

Comparision Results of 18 examination videos:
[https://bit.ly/UTR-Track-results](https://bit.ly/UTR-Track-results)

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

```bash
pip install -r requirements.txt

# PyTorch, matching your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU
# pip install torch torchvision                                                    # CPU
```

**About `boxmot`**: this repository vendors its own fork of BoxMOT under `boxmot/`,
which is where `UTRTrack` lives. The upstream `boxmot` package on PyPI does **not**
contain it. `utrtrack_pipeline.py` puts the vendored copy at the front of `sys.path`,
so the pipeline works from a plain clone with no extra step. If you import `boxmot`
from your own scripts, install the fork explicitly so it takes precedence:

```bash
pip install -e boxmot/
python -c "import boxmot; print(boxmot.__file__)"   # must point inside this repo
```

3. **ReID weights**:
```bash
# Place the OSNet-DCN weights in the repository root.
#   osnet_dcn_x0_5_endocv.pt   OSNet-DCN   (trained on IGH ReID)          -- default
#   osnet_dcn_x0_5_hybrid.pt   OSNet-DCN*  (IGH ReID + HyperKvasir ReID)
# Select one at run time with --reid_weights.
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
| `--model_weights` | *(none)* | Force one YOLO model for every video, bypassing organ-based selection. Required for datasets whose layout has no `UTTQ`/`UTDD` folders, such as Hyper-Kvasir |
| `--reid_weights` | `osnet_dcn_x0_5_endocv.pt` | ReID weights. `osnet_dcn_x0_5_endocv.pt` = OSNet-DCN, `osnet_dcn_x0_5_hybrid.pt` = OSNet-DCN* |
| `--output_dir` | **required** | Output directory. Convention: `content/exp_<id>_<var><value>`. Never reuse a directory from a previous experiment |
| `--tracker_type` | `tlukf` | Tracker type: `tlukf` (UTRTrack) or `xysr` (StrongSort) |
| `--iou_threshold` | `0.1` | IoU threshold for track matching |
| `--min_temporal_threshold` | `0` | Minimum temporal threshold (frames) |
| `--max_temporal_threshold` | `0` | Maximum temporal threshold (frames) |
| `--use_frame_id` | `False` | Use frame ID instead of timestamp |

### 5. Reproducing the paper results

Both appearance models reported in the paper are reproduced from this repository by
changing `--reid_weights` only; every other condition is held fixed.

```bash
# IGH, OSNet-DCN  -> Table "Detailed MOT metrics", OSNet-DCN rows of the comparison table
python utrtrack_pipeline.py --video_dir data/video_test --model_dir model_yolo     --reid_weights osnet_dcn_x0_5_endocv.pt --output_dir content/exp_igh_osnetdcn

# IGH, OSNet-DCN* -> Table "Per-sequence MOT metrics", OSNet-DCN* rows
python utrtrack_pipeline.py --video_dir data/video_test --model_dir model_yolo     --reid_weights osnet_dcn_x0_5_hybrid.pt --output_dir content/exp_igh_osnetdcn_hybrid

# Hyper-Kvasir: no UTTQ/UTDD layout, so the detector must be forced
python utrtrack_pipeline.py --video_dir data/hyper_kvasir/upper-gi-tract/valid     --model_weights model_yolo/thucquan.pt --output_dir content/exp_kvasir
```

The detector is otherwise chosen per organ from the video's directory path
(`UTTQ` -> `thucquan.pt`, `UTDD` -> `daday.pt`, else `htt.pt`). Every run prints
`Processing video: ... with model: ... | ReID: ...`; check that line matches the
organ before using a run for reported numbers.

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
    per_class=False,       # Cross-class tracking
    single_object_mode=True,      # Prefer re-associating the one tracked
                                   # lesion over spawning a new ID
    reacquire_min_iou=0.03,       # Single-object re-association: min IoU
    reacquire_max_cost=0.60,      # Single-object re-association: max appearance cost
    reacquire_max_age=30,         # Single-object re-association: max frames since last update
    max_virtual_age=5,            # Max consecutive frames a virtual box may be emitted for
    virtual_recent_hq_gap=4,      # Virtual box requires high-quality evidence within N frames
    virtual_conf=0.28,            # Base confidence for virtual (predicted) boxes
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