"""
Server implementations for FL.
"""

from .base_server import BaseServer
from .client_side_defense import *
from .anomaly_detection import *
from .robust_aggregation import *
from .fedavg_server import WeightedFedAvgServer, UnweightedFedAvgServer
from .augment_server import AugmentServer

__all__ = (
    ["BaseServer"]
    + robust_aggregation.__all__
    + anomaly_detection.__all__
    + client_side_defense.__all__,
    ["WeightedFedAvgServer", "UnweightedFedAvgServer"],
    ["AugmentServer"],
)
