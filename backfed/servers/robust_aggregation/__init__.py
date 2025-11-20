"""
Robust aggregation defenses modify the aggregation algorithm to be resilient
against malicious updates, typically by using robust statistics or downweighting malicious updates.

Examples: Median, Trimmed Mean, Krum, FoolsGold, etc.
"""

from .bulyan_server import BulyanServer
from .flare_server import FlareServer
from .fltrust_server import FLTrustServer
from .median_server import GeometricMedianServer, CoordinateMedianServer
from .foolsgold_server import FoolsGoldServer
from .krum_server import KrumServer, MultiKrumServer
from .trimmed_mean_server import TrimmedMeanServer
from .robustlr_server import RobustLRServer
from .weakdp_server import WeakDPServer, NormClippingServer
from .crfl_server import CRFLServer

__all__ = [
    "BulyanServer",
    "FLTrustServer", "FlareServer",
    "FoolsGoldServer",
    "KrumServer", "MultiKrumServer",
    "GeometricMedianServer", "CoordinateMedianServer",
    "TrimmedMeanServer",
    "RobustLRServer",
    "WeakDPServer", "NormClippingServer", "CRFLServer",
]
