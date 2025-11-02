"""
Anomaly detection defenses detect and filter out malicious client updates before aggregation.
Some defenses such as FLDetector permanently remove suspicious clients from training,
"""

from .anomaly_detection_server import AnomalyDetectionServer
from .deepsight_server import DeepSightServer
from .rflbat_server import RFLBATServer
from .flame_server import FlameServer
from .fldetector_server import FLDetectorServer
from .indicator_server import IndicatorServer
from .alignins_server import AlignInsServer
from .multi_metrics_server import MultiMetricsServer
from .snowball_server import SnowballServer
from .feddlad_server import FedDLADServer
from .ad_krum_server import ADMultiKrumServer

__all__ = [
    "AnomalyDetectionServer",
    "DeepSightServer",
    "RFLBATServer",
    "FlameServer",
    "FLDetectorServer",
    "IndicatorServer",
    "AlignInsServer",
    "MultiMetricsServer",
    "SnowballServer",
    "FedDLADServer",
    "ADMultiKrumServer",
]