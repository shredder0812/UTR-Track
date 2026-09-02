# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '15.0.3'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.hybridsort.hybridsort import HybridSort
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.strongsort.strongsort import StrongSort, StrongSortXYSR
from boxmot.trackers.utrtrack.utrtrack import UTRTrack

TRACKERS = [
    "bytetrack",
    "botsort",
    "strongsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "strongsortxysr",
    "utrtrack",
]

__all__ = (
    "__version__",
    "StrongSort",
    "StrongSortXYSR",
    "OcSort",
    "ByteTrack",
    "BotSort",
    "DeepOcSort",
    "HybridSort",
    "BoostTrack",
    "create_tracker",
    "get_tracker_config",
    "gsi",
    "UTRTrack"
)
