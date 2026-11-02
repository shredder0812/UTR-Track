"""Run UTR-Track-v0 (MBEC paper configuration) on one video or a folder of videos.

UTR-Track-v0 = class ``UTRTrack``: XYAH TL-UKF with hand-set process noise, two-tier
association, single-object mode, n_init = 1; YOLO at conf 0.45; ReID
``osnet_dcn_x0_5_endocv.pt``. Full parameter list: ``utrtrack_versions.py`` (UTRTRACK_V0).

Examples:
    python run_utrtrack_v0.py --video data/video_test --output_dir content/exp_v0_igh
    python run_utrtrack_v0.py --video data/video_test/UTDD/220324CS2_Trim.mp4 --output_dir content/exp_v0_220324CS2
"""

from utrtrack_pipeline import version_cli

if __name__ == "__main__":
    version_cli("utrtrack_v0")
