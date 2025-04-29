"""
Useful Constants and Typings.
"""

import torch
from typing import Dict, Tuple

NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "EMNIST_BYCLASS": 62,
    "EMNIST_BALANCED": 47,
    "EMNIST_DIGITS": 10,
    "FEMNIST": 62,
    "TINYIMAGENET": 200,
}

IMG_SIZE = {
    "CIFAR10": (32, 32, 3),
    "CIFAR100": (32, 32, 3),
    "MNIST": (28, 28, 1),
    "EMNIST_BYCLASS": (28, 28, 1),
    "EMNIST_BALANCED": (28, 28, 1),
    "EMNIST_DIGITS": (28, 28, 1),
    "FEMNIST": (28, 28, 1),
    "TINYIMAGENET": (64, 64, 3),
}

Metrics = Dict[str, float]
StateDict = Dict[str, torch.Tensor]
ClientTrainPackage = Tuple[int, StateDict, Metrics]
ClientEvalPackage = Tuple[int, Metrics]
