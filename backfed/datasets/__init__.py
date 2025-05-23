"""
Dataset implementations for FL.
"""

from backfed.datasets.sentiment140 import (
    load_sentiment140_for_albert, 
    sentiment140_collate_fn,
)
from backfed.datasets.femnist import FEMNIST
from backfed.datasets.reddit import (
    RedditCorpus,
    load_reddit_for_lstm
)

__all__ = [
    "Sentiment140Dataset",
    "FL_DataLoader",
    "FEMNIST",
    "load_sentiment140_for_albert",
    "RedditCorpus", 
    "load_reddit_for_lstm",
    "sentiment140_collate_fn",
]
