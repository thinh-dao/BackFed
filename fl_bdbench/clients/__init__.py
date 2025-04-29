"""
Client implementations for FL.
"""

from fl_bdbench.clients.base_client import BaseClient, ClientApp
from fl_bdbench.clients.base_benign_client import BenignClient
from fl_bdbench.clients.base_malicious_client import MaliciousClient
from fl_bdbench.clients.chameleon_malicious_client import ChameleonClient
from fl_bdbench.clients.neurotoxin_malicious_client import NeurotoxinClient
