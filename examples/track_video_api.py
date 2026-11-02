"""Minimal use of UTR-Track-v0 / UTR-Track-v1 from Python (no pipeline).

Runs YOLO + the chosen tracker on one video and writes the tracker's *raw* output
(no one-box-per-frame / ID-relabel step, unlike ``utrtrack_pipeline.py``) in MOT format:
``frame, id, x, y, w, h, score, cls, -1, -1``. Frames are 0-based, as in the ``mot/`` files of
``utrtrack_pipeline.py`` and the IGH ground truth.

Examples (from the repository root):
    python examples/track_video_api.py --version v1 --video data/video_test/UTDD/220324CS2_Trim.mp4 \
        --yolo model_yolo/daday.pt --out content/api_v1_220324CS2.txt
    python examples/track_video_api.py --version v0 --video data/video_test/UTTQ/230411BVK004_Trim.mp4 \
        --yolo model_yolo/thucquan.pt --out content/api_v0_230411BVK004.txt

Equivalent in your own code::

    from utrtrack_versions import get_version, build_tracker, preprocess_detections
    version = get_version("utrtrack_v1")
    tracker = build_tracker(version, reid_weights=None, device=torch.device("cuda:0"))
    dets = yolo(frame, conf=version.det_conf)[0].boxes.data.cpu().numpy()   # (N, 6)
    tracks = tracker.update(preprocess_detections(version, dets), frame)    # (M, 8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from utrtrack_versions import build_tracker, get_version, preprocess_detections  # noqa: E402


def track_video(version_name: str, video: Path, yolo_weights: Path, out_file: Path,
                reid_weights: str | None = None) -> int:
    """Track ``video`` and write raw tracker output to ``out_file``; return the number of rows."""
    version = get_version(version_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(yolo_weights))
    detector.fuse()
    tracker = build_tracker(version, reid_weights, device)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_rows, frame_idx = 0, -1
    with open(out_file, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1  # 0-based
            # Same input masking as utrtrack_pipeline.py (on-screen text of the endoscope)
            cv2.rectangle(frame, (0, 30), (220, 80), (255, 255, 255), -1)
            result = next(iter(detector(frame, stream=True, verbose=False, conf=version.det_conf)))
            dets = result.boxes.data.to("cpu").numpy()
            dets = preprocess_detections(version, dets)
            if dets.size == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(dets, frame)
            for x1, y1, x2, y2, tid, score, cls, _ in np.asarray(tracks).reshape(-1, 8):
                f.write(f"{frame_idx},{int(tid)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                        f"{score:.3f},{int(cls)},-1,-1\n")
                n_rows += 1
    cap.release()
    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=["v0", "v1"], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True,
                        help="YOLO weights (daday.pt for UTDD, thucquan.pt for UTTQ, htt.pt otherwise)")
    parser.add_argument("--out", type=Path, required=True, help="output MOT text file")
    parser.add_argument("--reid_weights", type=str, default=None,
                        help="default: osnet_dcn_x0_5_endocv.pt (v0) / osnet_dcn_x0_5_hybrid.pt (v1)")
    args = parser.parse_args()
    n = track_video(f"utrtrack_{args.version}", args.video, args.yolo, args.out, args.reid_weights)
    print(f"UTR-Track-{args.version}: {n} boxes written to {args.out}")


if __name__ == "__main__":
    main()
