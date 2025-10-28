"""
Client implementations for FL.
"""

from backfed.clients.base_client import BaseClient
from backfed.clients.base_benign_client import BenignClient
from backfed.clients.localdp_benign_client import LocalDPClient
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.clients.chameleon_malicious_client import ChameleonClient
from backfed.clients.neurotoxin_malicious_client import NeurotoxinClient
from backfed.clients.sentiment_malicious_client import SentimentMaliciousClient
from backfed.clients.reddit_malicious_client import RedditMaliciousClient
from backfed.clients.anticipate_malicious_client import AnticipateClient
from backfed.clients.cerberus_malicious_client import CerberusMaliciousClient

__all__ = [
    "BaseClient",
    "BenignClient",
    "LocalDPClient",
    "MaliciousClient",
    "ChameleonClient",
    "NeurotoxinClient",
    "SentimentMaliciousClient",
    "RedditMaliciousClient",
    "AnticipateClient",
    "CerberusMaliciousClient",
]