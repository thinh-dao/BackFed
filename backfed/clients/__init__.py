"""
Client implementations for FL.
"""

from backfed.clients.base_client import BaseClient, ClientApp
from backfed.clients.base_benign_client import BenignClient
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.clients.chameleon_malicious_client import ChameleonClient
from backfed.clients.neurotoxin_malicious_client import NeurotoxinClient
from backfed.clients.sentiment_malicious_client import SentimentMaliciousClient
from backfed.clients.reddit_malicious_client import RedditMaliciousClient
