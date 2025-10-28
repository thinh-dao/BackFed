"""
Model utility functions for FL.
"""

import torch
import torchvision
import backfed.models as models
import torchvision.transforms as transforms

# Import the LSTM model
from backfed.models.resnet_cifar import get_cifar_resnet_model
from backfed.models.resnet_mnist import get_mnist_resnet_model
from backfed.models.resnet_tinyimagenet import get_tinyimagenet_resnet_model
from backfed.models.vgg_cifar import get_cifar_vgg_model
from backfed.models.vgg_tinyimagenet import get_tinyimagenet_vgg_model
from backfed.models import get_lstm_model, get_albert_model
from backfed.utils.logging_utils import log
from logging import INFO

SUPPORT_MODELS = {
    'CIFAR10': ('resnet', 'vgg'),
    'CIFAR100': ('resnet', 'vgg'),
    'MNIST': ('mnistnet', 'resnet'),
    'EMNIST': ('mnistnet', 'resnet'),
    'FEMNIST': ('mnistnet', 'resnet'),
    'SENTIMENT140': ('albert',),
    'REDDIT': ('lstm',),
    'TINYIMAGENET': ('resnet', 'vgg'),
}


def _validate_model_choice(dataset_name: str, model_name: str) -> None:
    allowed_prefixes = SUPPORT_MODELS.get(dataset_name)
    if allowed_prefixes is None:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. Supported datasets: {list(SUPPORT_MODELS.keys())}"
        )
    if not any(model_name.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(
            f"Model {model_name} is not supported for dataset {dataset_name}. "
            f"Supported model prefixes: {allowed_prefixes}"
        )


def _build_cifar_model(model_name: str, num_classes: int):
    if model_name.startswith('resnet'):
        return get_cifar_resnet_model(model_name, num_classes)
    if model_name.startswith('vgg'):
        return get_cifar_vgg_model(model_name, num_classes=num_classes)
    raise ValueError(f"CIFAR model {model_name} is not supported.")


def _build_mnist_model(model_name: str, num_classes: int):
    if model_name == 'mnistnet':
        return models.MnistNet(num_classes)
    if model_name == 'resnet9':
        return get_mnist_resnet_model('resnet18', num_classes)
    if model_name.startswith('resnet'):
        return get_mnist_resnet_model(model_name, num_classes)
    raise ValueError(f"MNIST model {model_name} is not supported.")


def _replace_classifier(model, num_classes: int):
    if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    classifier = getattr(model, 'classifier', None)
    if isinstance(classifier, torch.nn.Linear):
        in_features = classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
        return model

    if isinstance(classifier, torch.nn.Sequential) and len(classifier) > 0:
        modules = list(classifier)
        for idx in range(len(modules) - 1, -1, -1):
            layer = modules[idx]
            if isinstance(layer, torch.nn.Linear):
                modules[idx] = torch.nn.Linear(layer.in_features, num_classes)
                model.classifier = torch.nn.Sequential(*modules)
                return model
    raise ValueError(f"Unable to replace classifier head for model type {type(model)}.")


def _load_torchvision_model(model_name: str, num_classes: int, weights=None):
    constructor = getattr(torchvision.models, model_name)
    model = constructor(weights=weights) if weights is not None else constructor()
    return _replace_classifier(model, num_classes)


def _build_tinyimagenet_model(model_name: str, num_classes: int, pretrain_model_path=None):
    if model_name == 'mnistnet':
        raise ValueError("MNISTNet is not supported for TINYIMAGENET dataset.")
    elif model_name.startswith('resnet') and pretrain_model_path is None:
        return get_tinyimagenet_resnet_model(model_name, num_classes)
    elif model_name.startswith('vgg') and pretrain_model_path is None:
        return get_tinyimagenet_vgg_model(model_name, num_classes=num_classes)

    if pretrain_model_path is None:
        constructor = getattr(torchvision.models, model_name)
        return constructor(num_classes=num_classes)
    if pretrain_model_path in {"IMAGENET1K_V1", "IMAGENET1K_V2"}:
        log(INFO, f"Load pretrained model from {pretrain_model_path}")
        return _load_torchvision_model(model_name, num_classes, weights=pretrain_model_path)

    constructor = getattr(torchvision.models, model_name)
    model = constructor()
    state_dict = torch.load(pretrain_model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return _replace_classifier(model, num_classes)


def _build_sentiment_model(model_name: str, num_classes: int, dataset_name: str):
    if model_name != 'albert':
        raise ValueError(f"Model {model_name} is not supported for {dataset_name} dataset.")
    model = get_albert_model(dataset_name=dataset_name, num_classes=num_classes)
    log(INFO, f"Created Albert model for {dataset_name}")
    return model


def _build_reddit_model(model_name: str, num_classes: int):
    if model_name != 'lstm':
        raise ValueError("Only 'lstm' model is supported for REDDIT dataset.")
    model = get_lstm_model('reddit', num_tokens=num_classes)
    log(INFO, "Created LSTM model for REDDIT")
    return model


DATASET_BUILDERS = {
    'CIFAR': _build_cifar_model,
    'CIFAR10': _build_cifar_model,
    'CIFAR100': _build_cifar_model,
    'MNIST': _build_mnist_model,
    'EMNIST': _build_mnist_model,
    'FEMNIST': _build_mnist_model,
    'TINYIMAGENET': _build_tinyimagenet_model,
    'SENTIMENT140': _build_sentiment_model,
    'REDDIT': _build_reddit_model,
}


def get_model(model_name, num_classes, dataset_name, pretrain_model_path=None):
    """Return a torchvision model with the given name and number of classes."""
    model_name = model_name.lower()
    dataset_name = dataset_name.upper()

    _validate_model_choice(dataset_name, model_name)

    builder = DATASET_BUILDERS.get(dataset_name)
    if builder is None:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. Supported datasets: {list(SUPPORT_MODELS.keys())}"
        )

    if dataset_name == 'TINYIMAGENET':
        return builder(model_name, num_classes, pretrain_model_path=pretrain_model_path)
    if dataset_name == 'SENTIMENT140':
        return builder(model_name, num_classes, dataset_name=dataset_name)

    return builder(model_name, num_classes)

def get_layer_names(model_name: str):
    if model_name.lower() == "mnistnet":
        model = models.MnistNet(num_classes=1)
    else:
        model = getattr(torchvision.models, model_name.lower())(num_classes=1)
    return list(model.state_dict().keys())

def get_normalization(dataset_name: str):
    """Normalization is separated so that the trigger pattern can be normalized."""
    dataset = dataset_name.upper()
    if "NIST" in dataset:
        return transforms.Normalize(mean=[0.1307], std=[0.3081])
    elif dataset in ["CIFAR10", "CIFAR100"]:
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    elif dataset == "TINYIMAGENET":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

def get_last_layer_name(model):
    last_layer_name = None
    for name, _ in model.named_modules():
        last_layer_name = name
    return last_layer_name
