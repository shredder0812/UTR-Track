"""Official configurations of UTR-Track-v0 and UTR-Track-v1.

Single source of truth for everything that differs between the two versions:
the tracker class and its parameters, the YOLO confidence threshold, the
detection pre-processing, and the default ReID weights. ``utrtrack_pipeline.py``,
``run_utrtrack_v0.py``, ``run_utrtrack_v1.py`` and ``examples/`` all read from here.

    v0  UTRTrack    XYAH TL-UKF, hand-set Q, single-object mode, n_init=1   (MBEC paper)
    v1  UTRTrackV1  XYSR TL-UKF, calibrated Q, 3-tier association, n_init=3,
                    re-acquisition, birth control, lost-track absorption,
                    no single-object mode                                   (thesis)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Use the BoxMOT fork vendored in this repository (it carries UTRTrack / UTRTrackV1).
_VENDORED_BOXMOT = Path(__file__).resolve().parent / "boxmot"
if (_VENDORED_BOXMOT / "boxmot" / "__init__.py").is_file() and str(_VENDORED_BOXMOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_BOXMOT))

from boxmot import UTRTrack, UTRTrackV1  # noqa: E402  (must follow sys.path setup)


@dataclass(frozen=True)
class UTRTrackVersion:
    """Everything needed to run one UTR-Track version.

    Attributes:
        name: version key (``utrtrack_v0`` / ``utrtrack_v1``).
        label: human-readable name.
        tracker_class: BoxMOT tracker class.
        det_conf: YOLO confidence threshold.
        class_agnostic_nms_iou: IoU of the class-agnostic NMS applied to detections
            before tracking (``None`` = not applied).
        reid_weights: default ReID weights file.
        tracker_kwargs: constructor arguments of ``tracker_class`` (besides ReID/device).
    """

    name: str
    label: str
    tracker_class: type
    det_conf: float
    class_agnostic_nms_iou: float | None
    reid_weights: str
    tracker_kwargs: dict[str, Any]


UTRTRACK_V0 = UTRTrackVersion(
    name="utrtrack_v0",
    label="UTR-Track-v0",
    tracker_class=UTRTrack,
    det_conf=0.45,
    class_agnostic_nms_iou=None,
    reid_weights="osnet_dcn_x0_5_endocv.pt",
    tracker_kwargs=dict(
        max_cos_dist=0.9,
        max_iou_dist=0.9,
        max_age=500,
        n_init=1,
        ema_alpha=0.95,
        mc_lambda=0.98,
        per_class=False,
        single_object_mode=True,
        reacquire_min_iou=0.03,
        reacquire_max_cost=0.60,
        reacquire_max_age=30,
        max_virtual_age=5,
        virtual_recent_hq_gap=4,
        virtual_conf=0.28,
    ),
)

UTRTRACK_V1 = UTRTrackVersion(
    name="utrtrack_v1",
    label="UTR-Track-v1",
    tracker_class=UTRTrackV1,
    det_conf=0.1,
    class_agnostic_nms_iou=0.45,
    reid_weights="osnet_dcn_x0_5_hybrid.pt",
    tracker_kwargs=dict(
        max_cos_dist=0.9,
        max_iou_dist=0.9,
        max_age=500,
        n_init=3,
        ema_alpha=0.95,
        mc_lambda=0.98,
        per_class=False,
        reacquire_min_iou=0.01,
        reacquire_max_cost=0.75,
        reacquire_max_age=500,
        max_virtual_age=5,
        virtual_recent_hq_gap=4,
        virtual_conf=0.28,
        q_calibration_scale=1.0,
        absorption_iou=0.3,
        absorption_app_sim=0.7,
        birth_iou=0.3,
        birth_contain=0.8,
    ),
)

# "tlukf" is the historical name of v0 in utrtrack_pipeline.py.
VERSIONS: dict[str, UTRTrackVersion] = {
    "utrtrack_v0": UTRTRACK_V0,
    "utrtrack_v1": UTRTRACK_V1,
    "tlukf": UTRTRACK_V0,
}


def get_version(name: str) -> UTRTrackVersion:
    """Return the configuration registered under ``name`` (``utrtrack_v0``, ``utrtrack_v1`` or ``tlukf``)."""
    try:
        return VERSIONS[name]
    except KeyError:
        raise ValueError(f"Unknown UTR-Track version {name!r}; choose one of {sorted(VERSIONS)}") from None


def build_tracker(version: UTRTrackVersion, reid_weights: str | Path | None, device, half: bool = False):
    """Instantiate the tracker of ``version``.

    Args:
        version: configuration from :data:`VERSIONS`.
        reid_weights: ReID weights path; ``None`` uses ``version.reid_weights``.
        device: ``torch.device`` for the ReID model.
        half: fp16 ReID inference.
    """
    weights = Path(reid_weights) if reid_weights is not None else Path(version.reid_weights)
    return version.tracker_class(reid_weights=weights, device=device, half=half, **version.tracker_kwargs)


def class_agnostic_nms(det_boxes: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS across classes on ``(N, 6)`` detections ``[x1, y1, x2, y2, conf, cls]``.

    YOLO's NMS runs per class, so two almost identical boxes with different labels on
    the same lesion both reach the tracker and can start two competing tracks. Keeping
    only the highest-confidence box of every group with ``IoU > iou_thresh`` removes
    them; boxes of lesions at different locations (low IoU) are never merged.
    Kept boxes are returned in their original order.
    """
    if det_boxes is None or len(det_boxes) == 0:
        return det_boxes
    boxes = np.asarray(det_boxes, dtype=np.float64)
    order = np.argsort(-boxes[:, 4])
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_r - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thresh]
    return det_boxes[np.array(sorted(keep))]


def preprocess_detections(version: UTRTrackVersion, det_boxes: np.ndarray) -> np.ndarray:
    """Apply the detection pre-processing of ``version`` (class-agnostic NMS for v1)."""
    if version.class_agnostic_nms_iou is not None:
        return class_agnostic_nms(det_boxes, version.class_agnostic_nms_iou)
    return det_boxes
