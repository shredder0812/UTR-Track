# UTR-Track

## Overview

**UTR-Track** (Universal Tracking with Transfer Learning) is a lesion detection-and-tracking
system for upper gastrointestinal endoscopy video. YOLOv8 detects lesions and a
StrongSORT-style tracker built around a dual Source/Primary **Transfer-Learning Unscented
Kalman Filter (TL-UKF)** with **Virtual Trajectory Augmentation (VTA)** follows them.
Appearance features come from OSNet-DCN.

The repository contains two versions of the tracker:

| Version | Class | Where it was reported |
|---|---|---|
| **UTR-Track-v0** | `UTRTrack` (alias `UTRTrackV0`) | MBEC paper |
| **UTR-Track-v1** | `UTRTrackV1` | Master's thesis (current version) |

### Tracking Results

Comparison results of 18 examination videos (UTR-Track-v0):
[https://bit.ly/UTR-Track-results](https://bit.ly/UTR-Track-results)

### Key Features

- **Multi-Pathology Detection**: esophagitis, esophageal cancer, gastritis, gastric cancer, duodenal ulcer
  (one YOLO model per organ, selected from the video's folder).
- **TL-UKF tracking**: two filters per track. The Source filter only takes
  high-confidence detections, and the Primary filter learns from it when the lesion is missed.
- **Virtual trajectory boxes** for short detection gaps.
- **Cross-class tracking** (`per_class=False`): a lesion keeps its identity when the detector changes its label.
- Camera-motion compensation (ECC) and appearance-based re-identification.

## UTR-Track-v0 and UTR-Track-v1

### What changes from v0 to v1

| | UTR-Track-v0 | UTR-Track-v1 |
|---|---|---|
| Tracker class | `UTRTrack` (`boxmot/boxmot/trackers/utrtrack/`) | `UTRTrackV1` (`boxmot/boxmot/trackers/utrtrack_v1/`) |
| Kalman state | XYAH `[x, y, w/h, h]` (`tlukf.py`) | XYSR `[x, y, area, w/h]` (`xysr_tlukf_v1.py`) |
| Process noise `Q` | hand-set | calibrated from ground-truth accelerations (`q_calibration_scale=1`) |
| YOLO confidence | 0.45 | 0.1 |
| Detection pre-processing | – | class-agnostic NMS, IoU 0.45 |
| Association | appearance cascade + IoU | appearance cascade + IoU + low-confidence tier (detections 0.1–0.45, IoU only, never start a track) |
| Track confirmation `n_init` | 1 | 3 |
| Lost-track recovery | one global best pair, single-object mode only | Hungarian re-acquisition (IoU ≥ 0.01 or appearance distance ≤ 0.75) |
| New detections overlapping a track | delete the old track and start a new id | **birth control**: a duplicate of a matched track is ignored, and a detection on a lost track updates it with its old id |
| Two tracks on one lesion | – | **lost-track absorption**: the matched track inherits the older id |
| Default ReID weights | `osnet_dcn_x0_5_endocv.pt` (OSNet-DCN) | `osnet_dcn_x0_5_hybrid.pt` (OSNet-DCN*) |

Both versions share: OSNet-DCN appearance, ECC camera-motion compensation, dual
Source/Primary TL-UKF with transfer learning, VTA (at most 5 miss steps, so 2 virtual
boxes per gap in v1), `max_age=500`, `max_cos_dist=max_iou_dist=0.9`, `mc_lambda=0.98`,
`ema_alpha=0.95`, and the same output layer of `utrtrack_pipeline.py` (see [Output](#output)).

All parameter values live in one place: [utrtrack_versions.py](utrtrack_versions.py)
(`UTRTRACK_V0`, `UTRTRACK_V1`). The pipeline, the run scripts and the examples read them from there.

### Results on IGH (18 sequences)

Results were computed with `py-motmetrics` (`eval_motchallenge.py`, protocol `mot15-2D`,
IoU 0.5) on the files written to `output_dir/mot/`. Each configuration was run once.

| Version | ReID weights | IDF1 | MOTA | IDs | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| UTR-Track-v1 | hybrid (default) | 85.4 | 62.4 | 0 | 984 | 1369 |
| UTR-Track-v0 | hybrid | 74.0 | 45.2 | 0 | 952 | 2475 |
| UTR-Track-v0 | endocv (default) | 73.7 | 44.6 | 0 | 972 | 2495 |

Caveats:
- The thesis lists 73.8 IDF1 for v0 with the endocv weights (same FP/FN/MOTA); the 0.1 gap
  to this run has not been traced.
- The IGH labels contain 283 boxes that share an identity with another box in the same
  frame. This inflates IDF1 for every method; removing them lowers v1's IDF1 by 3.2–5.0 points.
- Each IGH sequence contains a single lesion, so `IDs` only measures whether one identity
  is split, not whether two lesions are confused.
- These are technical results on retrospective data, not a clinical validation.

### Verification of the v1 port

`UTRTrackV1` is a cleaned port of the research code (`UTRTrackXYSRv7` with E1, E4, the E6
re-acquisition thresholds, E6 absorption and E7c birth control) from the development
repository. Experimental branches that v1 does not use (flow-guided UKF, correlated TL,
quality-adaptive noise, ambiguity gating, track revival, consolidation, hysteresis,
vacuous gate) were removed. The numerical code was copied without reordering the operations.

Check performed on 2026-11-02:
- Command: `python run_utrtrack_v1.py --video data/video_test` on the 20 IGH test videos.
- Result: all 20 files in `mot/` are **byte-identical** to the development run that produced the
  thesis numbers, and `eval_motchallenge.py` gives the row above (IDF1 85.4, MOTA 62.4, IDs 0,
  FP 984, FN 1369).
- Environment: Python 3.10.19, PyTorch 2.3.1+cu118, Ultralytics 8.4.15, NumPy 1.26.4,
  OpenCV 4.9.0, RTX 4070.
- `run_utrtrack_v0.py --reid_weights osnet_dcn_x0_5_hybrid.pt` reproduces the development
  run of v0 (E0) byte for byte, in the same environment.

Other library versions or GPUs can change detector and ReID outputs slightly, and therefore
the metrics.

## Project Structure

```
UTR-Track/
├── utrtrack_pipeline.py           # Main pipeline: detect -> track -> video / CSV / MOT files
├── utrtrack_versions.py           # Official v0 / v1 configurations, tracker factory, class-agnostic NMS
├── run_utrtrack_v0.py             # Run UTR-Track-v0 (paper configuration)
├── run_utrtrack_v1.py             # Run UTR-Track-v1 (thesis configuration)
├── examples/
│   └── track_video_api.py         # Use v0 / v1 directly from Python (raw tracker output)
├── osnet_dcn_x0_5_endocv.pt       # ReID weights, OSNet-DCN  (default of v0)
├── osnet_dcn_x0_5_hybrid.pt       # ReID weights, OSNet-DCN* (default of v1)
├── model_yolo/                    # YOLO models
│   ├── daday.pt                   # stomach      (videos under .../UTDD/...)
│   ├── thucquan.pt                # esophagus    (videos under .../UTTQ/...)
│   └── htt.pt                     # duodenum     (everything else)
├── data/video_test/               # Input videos
└── boxmot/                        # Vendored BoxMOT fork (AGPL-3.0)
    └── boxmot/
        ├── trackers/
        │   ├── utrtrack/          # UTR-Track-v0: utrtrack.py + sort/{track,tracker,linear_assignment,...}
        │   ├── utrtrack_v1/       # UTR-Track-v1: utrtrack_v1.py + sort/{track,tracker}
        │   └── strongsort/ ...    # Baseline trackers
        ├── motion/kalman_filters/aabb/
        │   ├── tlukf.py           # XYAH TL-UKF (v0)
        │   └── xysr_tlukf_v1.py   # XYSR TL-UKF with calibrated Q (v1)
        └── configs/trackers/
            ├── utrtrack.yaml      # v0 defaults for create_tracker("utrtrack")
            └── utrtrackv1.yaml    # v1 defaults for create_tracker("utrtrackv1")
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
which is where `UTRTrack` and `UTRTrackV1` live. The upstream `boxmot` package on PyPI
does **not** contain them. `utrtrack_pipeline.py` and `utrtrack_versions.py` put the
vendored copy at the front of `sys.path`, so they work from a plain clone. If you import
`boxmot` from your own scripts, install the fork explicitly so it takes precedence:

```bash
pip install -e boxmot/
python -c "import boxmot; print(boxmot.__file__)"   # must point inside this repo
```

3. **ReID weights** (repository root):
```bash
#   osnet_dcn_x0_5_endocv.pt   OSNet-DCN   (trained on IGH ReID)                 -- default of v0
#   osnet_dcn_x0_5_hybrid.pt   OSNet-DCN*  (IGH ReID + HyperKvasir ReID)         -- default of v1
# Override with --reid_weights.
```

4. **YOLO models**:
```bash
mkdir -p model_yolo
# Copy or train models: daday.pt, thucquan.pt, htt.pt
```

## Usage

### 1. Run UTR-Track-v1 or UTR-Track-v0

`--video` accepts one video file or a folder (searched recursively).

```bash
# UTR-Track-v1 (thesis configuration)
python run_utrtrack_v1.py --video data/video_test --output_dir content/exp_v1_igh
python run_utrtrack_v1.py --video data/video_test/UTDD/220324CS2_Trim.mp4 --output_dir content/exp_v1_220324CS2

# UTR-Track-v0 (paper configuration)
python run_utrtrack_v0.py --video data/video_test --output_dir content/exp_v0_igh
```

The run scripts take `--model_dir`, `--model_weights` and `--reid_weights`. Every other
setting is fixed by the version.

### 2. Full pipeline

```bash
python utrtrack_pipeline.py \
    --video_dir data/video_test \
    --model_dir model_yolo \
    --output_dir content/exp_v1_igh \
    --tracker_type utrtrack_v1
```

`--tracker_type`:

| Value | Tracker |
|---|---|
| `utrtrack_v1` | UTR-Track-v1 |
| `utrtrack_v0` | UTR-Track-v0 |
| `tlukf` | UTR-Track-v0 (historical name, **default**, kept so that existing commands keep their meaning) |
| `xysr` | StrongSortXYSR baseline |

### 3. Parameter Details

| Parameter | Default | Description |
|---------|----------|-------|
| `--video_dir` | `data/video_test` | Video file, or directory containing input videos |
| `--model_dir` | `model_yolo` | Directory containing YOLO models |
| `--model_weights` | *(none)* | Force one YOLO model for every video, bypassing organ-based selection. Required for datasets whose layout has no `UTTQ`/`UTDD` folders, such as Hyper-Kvasir |
| `--reid_weights` | per version | `osnet_dcn_x0_5_endocv.pt` (v0, xysr) or `osnet_dcn_x0_5_hybrid.pt` (v1) |
| `--output_dir` | **required** | Output directory. Convention: `content/exp_<id>_<var><value>`. Never reuse a directory from a previous experiment |
| `--tracker_type` | `tlukf` | `utrtrack_v1`, `utrtrack_v0`, `tlukf` (= v0) or `xysr` |
| `--iou_threshold` | `0.1` | IoU threshold of the output ID relabelling |
| `--min_temporal_threshold` | `0` | Minimum temporal threshold (frames) |
| `--max_temporal_threshold` | `0` | Maximum temporal threshold (frames) |
| `--use_frame_id` | `False` | Use frame ID instead of timestamp |

### 4. Reproducing reported results

```bash
# UTR-Track-v1, IGH (thesis): OSNet-DCN* is the v1 default
python run_utrtrack_v1.py --video data/video_test --output_dir content/exp_v1_igh

# UTR-Track-v1, Hyper-Kvasir: no UTTQ/UTDD layout, so the detector must be forced
python run_utrtrack_v1.py --video data/hyper_kvasir/upper-gi-tract/valid \
    --model_weights model_yolo/thucquan.pt --output_dir content/exp_v1_kvasir

# UTR-Track-v0, IGH, OSNet-DCN  -> paper, "Detailed MOT metrics" / OSNet-DCN rows
python utrtrack_pipeline.py --tracker_type utrtrack_v0 --video_dir data/video_test --model_dir model_yolo \
    --reid_weights osnet_dcn_x0_5_endocv.pt --output_dir content/exp_igh_osnetdcn

# UTR-Track-v0, IGH, OSNet-DCN* -> paper, "Per-sequence MOT metrics" / OSNet-DCN* rows
python utrtrack_pipeline.py --tracker_type utrtrack_v0 --video_dir data/video_test --model_dir model_yolo \
    --reid_weights osnet_dcn_x0_5_hybrid.pt --output_dir content/exp_igh_osnetdcn_hybrid

# UTR-Track-v0, Hyper-Kvasir
python utrtrack_pipeline.py --tracker_type utrtrack_v0 --video_dir data/hyper_kvasir/upper-gi-tract/valid \
    --model_weights model_yolo/thucquan.pt --output_dir content/exp_kvasir
```

The detector is otherwise chosen per organ from the video's directory path
(`UTTQ` -> `thucquan.pt`, `UTDD` -> `daday.pt`, else `htt.pt`). Every run prints
`Processing video: ... with model: ... | Tracker: ... | ReID: ...`. Check that this line
matches the organ before using a run for reported numbers.

### 5. Python API

```python
import torch
from utrtrack_versions import get_version, build_tracker, preprocess_detections

version = get_version("utrtrack_v1")                    # or "utrtrack_v0"
tracker = build_tracker(version, reid_weights=None, device=torch.device("cuda:0"))
dets = yolo(frame, conf=version.det_conf)[0].boxes.data.cpu().numpy()   # (N, 6) x1,y1,x2,y2,conf,cls
tracks = tracker.update(preprocess_detections(version, dets), frame)    # (M, 8) x1,y1,x2,y2,id,score,cls,det_ind
```

A complete script is in [examples/track_video_api.py](examples/track_video_api.py).
It writes the **raw** tracker output, without the pipeline's output layer. The classes can
also be built directly (`from boxmot import UTRTrackV0, UTRTrackV1`; the constructor
defaults of `UTRTrackV1` are the v1 configuration) or through
`boxmot.create_tracker("utrtrack" | "utrtrackv1", ...)`.

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

**Output layer of the pipeline (both versions).** After the tracker, `utrtrack_pipeline.py`
keeps **one box per frame** (the highest-scoring real box, else the best virtual box), then
**relabels its id** with the id of the previous output box when they have the same class and
IoU > `--iou_threshold`. The reported v0/v1 numbers come from these files. Raw tracker output
(several tracks per frame) is available through the Python API. For v1 on IGH, the raw output
also has no ID switch (IDF1 85.3, MOTA 62.2).

In the MOT file the seventh column holds the track id, not a confidence score
(kept unchanged for compatibility with earlier evaluations).

### CSV Output Format:

```csv
timestamp_hms,timestamp_hmsf,frame_idx,fps,object_cls,object_idx,object_id,notes,
frame_height,frame_width,scale_height,scale_width,x1,y1,x2,y2,center_x,center_y
```

### Box labels (`notes` column, prefix on the video)

The label depends only on the box score:

- **Tracking**: score >= 0.6
- **Tracking Low Confidence**, prefix **(L)**: 0.45 <= score < 0.6
- **Virtual**, prefix **(V)**: score < 0.45

In v1 the **(V)** group also contains *real* boxes matched from weak detections
(0.1–0.45). A virtual (VTA) box always scores < 0.35: 0.238 for the first missed
frame and 0.172 for the second in v1.

## 🔧 Advanced Configuration

Edit the version entries in [utrtrack_versions.py](utrtrack_versions.py), not the pipeline:

```python
UTRTRACK_V1 = UTRTrackVersion(
    name="utrtrack_v1",
    tracker_class=UTRTrackV1,
    det_conf=0.1,                    # YOLO confidence threshold
    class_agnostic_nms_iou=0.45,     # detection NMS across classes (None = off)
    reid_weights="osnet_dcn_x0_5_hybrid.pt",
    tracker_kwargs=dict(
        max_cos_dist=0.9, max_iou_dist=0.9, max_age=500, n_init=3,
        ema_alpha=0.95, mc_lambda=0.98, per_class=False,
        reacquire_min_iou=0.01, reacquire_max_cost=0.75, reacquire_max_age=500,
        max_virtual_age=5, virtual_recent_hq_gap=4, virtual_conf=0.28,
        q_calibration_scale=1.0,                  # None = hand-set Q of v0
        absorption_iou=0.3, absorption_app_sim=0.7,
        birth_iou=0.3, birth_contain=0.8,         # birth_contain=None = IoU test only
    ),
)
```

`UTRTRACK_V0` follows the same pattern (`n_init=1`,
`reacquire_min_iou=0.03`, `reacquire_max_cost=0.60`, `reacquire_max_age=30`, YOLO conf 0.45).
Changing any value changes the configuration: give the run a new name and do not report it as v0/v1.

The miss counter of v1 tracks grows by two per missed frame, so `max_age=500` keeps a lost
track for about 250 frames, and `max_virtual_age=5` allows two virtual boxes per gap.

**StrongSort (XYSR)** baseline (`--tracker_type xysr`) is configured in `ObjectDetection._initialize_tracker()`:
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
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864) (low-confidence association tier of v1)
- Cao, Thanh-Tung, et al. (2025). *A Combination of YOLO and OSNet Re-ID Neuronal Networks for Tracking Abnormalities in Upper Gastrointestinal Endoscopy Videos*. In **Information and Communication Technology**, pp. 55–68. Springer Nature Singapore. ISBN: 978-981-96-4285-4.
- Cao, Thanh-Tung, et al. (2025). *OSNet-DCN: Integrating Deformable Feature Learning for Tracking Lesion in Endoscopic Videos*. In **2025 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)**, pp. 1–6. https://doi.org/10.1109/MAPR67746.2025.11134015
- Cao, Thanh-Tung, et al. (2026). *UTR-Track: Unscented Trajectory Reconstruction with Re-identification for Robust Lesion Tracking in Endoscopy Videos*. **Medical & Biological Engineering & Computing**. Accepted manuscript; currently in the publication process. Manuscript number: MBEC-D-26-01692R3.


## Contact

For questions or issues, please contact: [tung.ct242509m@sis.hust.edu.vn]
