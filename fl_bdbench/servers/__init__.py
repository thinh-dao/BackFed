"""
Server implementations for FL.
"""

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.servers.defense_categories import (
    ClientSideDefenseServer,
    RobustAggregationServer,
    AnomalyDetectionServer,
    PostAggregationServer,
)
from fl_bdbench.servers.trimmed_mean_server import TrimmedMeanServer
from fl_bdbench.servers.median_server import GeometricMedianServer, CoordinateMedianServer
from fl_bdbench.servers.multi_krum_server import MultiKrumServer, KrumServer
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer, WeightedFedAvgServer
from fl_bdbench.servers.fedprox_server import FedProxServer
from fl_bdbench.servers.flame_server import FlameServer
from fl_bdbench.servers.foolsgold_server import FoolsGoldServer
from fl_bdbench.servers.weakdp_server import WeakDPServer, NormClippingServer
from fl_bdbench.servers.deepsight_server import DeepSightServer
from fl_bdbench.servers.rflbat_server import RFLBATServer
from fl_bdbench.servers.fldetector_server import FLDetectorServer
from fl_bdbench.servers.fltrust_server import FLTrustServer
from fl_bdbench.servers.flare_server import FlareServer
from fl_bdbench.servers.robustlr_server import RobustLRServer

__all__ = [
    # Base classes
    "BaseServer",
    "ClientSideDefenseServer",
    "RobustAggregationServer",
    "AnomalyDetectionServer",
    "PostAggregationServer",

    # Server implementations
    "TrimmedMeanServer",
    "GeometricMedianServer",
    "CoordinateMedianServer",
    "MultiKrumServer",
    "KrumServer",
    "UnweightedFedAvgServer",
    "WeightedFedAvgServer",
    "FedProxServer",
    "FlameServer",
    "FoolsGoldServer",
    "NormClippingServer",
    "WeakDPServer",
    "DeepSightServer",
    "RFLBATServer",
    "FLDetectorServer",
    "FLTrustServer",
    "FlareServer",
    "RobustLRServer"
]
