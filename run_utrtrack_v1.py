"""Run UTR-Track-v1 (thesis configuration) on one video or a folder of videos.

UTR-Track-v1 = class ``UTRTrackV1``: XYSR TL-UKF with calibrated process noise,
three-tier association (low-confidence tier), n_init = 3, re-acquisition,
identity-preserving birth control and lost-track absorption, no single-object mode;
YOLO at conf 0.1 followed by class-agnostic NMS (IoU 0.45); ReID
``osnet_dcn_x0_5_hybrid.pt``. Full parameter list: ``utrtrack_versions.py`` (UTRTRACK_V1).

Examples:
    python run_utrtrack_v1.py --video data/video_test --output_dir content/exp_v1_igh
    python run_utrtrack_v1.py --video data/video_test/UTDD/220324CS2_Trim.mp4 --output_dir content/exp_v1_220324CS2
    # Datasets without UTTQ/UTDD folders (e.g. Hyper-Kvasir) need an explicit detector:
    python run_utrtrack_v1.py --video data/hyper_kvasir/upper-gi-tract/valid \
        --model_weights model_yolo/thucquan.pt --output_dir content/exp_v1_kvasir
"""

from utrtrack_pipeline import version_cli

if __name__ == "__main__":
    version_cli("utrtrack_v1")
