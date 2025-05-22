"""
Model utility functions for FL.
"""

import torch
import torchvision
import backfed.models as models
import torchvision.transforms as transforms

# Import the LSTM model
from backfed.models import get_lstm_model, get_transformer_model, get_albert_model
from backfed.utils.logging_utils import log
from logging import INFO

def get_model(model_name, num_classes, dataset_name, pretrain_model_path=None):
    """Return a torchvision model with the given name and number of classes."""
    model_name = model_name.lower()
    dataset_name = dataset_name.upper()

    # By default, use SimpleNet for MNIST
    if "NIST" in dataset_name:
        if model_name == "resnet9":
            model = models.mnist_resnet9(num_classes)
        elif model_name == "resnet18":
            model = models.mnist_resnet18(num_classes)
        elif model_name == "resnet34":
            model = models.mnist_resnet34(num_classes)
        elif model_name == "resnet50":
            model = models.mnist_resnet50(num_classes)
        elif model_name == "mnistnet":
            model = models.MnistNet(num_classes)
            
    elif dataset_name in ["CIFAR10", "CIFAR100", "TINYIMAGENET",]:
        if model_name == "mnistnet":
            raise ValueError(f"MNISTNet is not supported for {dataset_name} dataset.")
        if pretrain_model_path == None:
            model = getattr(torchvision.models, model_name)(num_classes=num_classes)
        elif pretrain_model_path == "IMAGENET1K_V1" or pretrain_model_path == "IMAGENET1K_V2":
            log(INFO, f"Load pretrained model from {pretrain_model_path}")
            model = getattr(torchvision.models, model_name)(weights=pretrain_model_path)
            # Replace the last fully connected layer
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
        else:
            model = getattr(torchvision.models, model_name)
            model.load_state_dict(torch.load(pretrain_model_path))
            
    elif dataset_name == 'SENTIMENT140':
        if model_name != "albert":
            raise ValueError(f"Model {model_name} is not supported for {dataset_name} dataset.")
        model = get_albert_model(dataset_name=dataset_name, num_classes=num_classes)
        log(INFO, f"Created Albert model for {dataset_name}")
    
    elif dataset_name == 'REDDIT':
        if model_name != "lstm":
            raise ValueError(f"Model {model_name} is not supported for {dataset_name} dataset.")
        model = get_lstm_model(dataset_name.lower(), num_tokens=num_classes) 
        log(INFO, f"Created LSTM model for {dataset_name}")
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return model

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
        return transforms.Normalize(mean=[0.5], std=[0.5])
    elif dataset in ["CIFAR10", "CIFAR100"]:
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    elif dataset == "TINYIMAGENET":
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

def get_last_layer_name(model):
    last_layer_name = None
    for name, _ in model.named_modules():
        last_layer_name = name
    return last_layer_name
