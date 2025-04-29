"""
Server utilities for FL.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

def test(model, test_loader, device, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
    """Validate the model performance on the test set."""
    model.eval()
    model.to(device)
    correct, loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if normalization:
                inputs = normalization(inputs)
                
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss += loss_fn(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total_samples += len(inputs)
    accuracy = correct / total_samples
    loss = loss / len(test_loader)
    return loss, accuracy

### Aggregation utility functions
def clip_updates_inplace(delta_weights_per_client: Dict[str, torch.Tensor], clipping_norm: float):
    """Clip the model updates within the clipping norm in place"""
    for client_weights in delta_weights_per_client:
        flatten_weights = torch.cat([weight.flatten() for weight in client_weights.values()])
        weight_diff_norm = torch.norm(flatten_weights, p=2)
        scaling_factor = min(1, clipping_norm / weight_diff_norm)
        
        for param in client_weights.values():
            param.mul_(scaling_factor)

def clip_updates(delta_weights_per_client: Dict[str, torch.Tensor], clipping_norm: float):
    """Clip the model updates within the clipping norm"""
    clipped_delta_weights_per_client = copy.deepcopy(delta_weights_per_client)
    for client_weights in clipped_delta_weights_per_client:
        flatten_weights = torch.cat([weight.flatten() for weight in client_weights.values()])
        weight_diff_norm = torch.norm(flatten_weights, p=2)
        scaling_factor = min(1, clipping_norm / weight_diff_norm)
        
        for param in client_weights.values():
            param.mul_(scaling_factor)

    return clipped_delta_weights_per_client

def cos_sim(client_params: Dict[str, torch.Tensor], global_params: Dict[str, torch.Tensor]):
    """ 
    Calculate the cosine similarity between the weight vectors of the client and global parameters.
    Args:
        client_params: the client parameters (Dict[str, torch.Tensor])
        global_params: the global parameters (Dict[str, torch.Tensor])
    Returns:
        sim: the cosine similarity between the client and global parameters (torch.tensor)
    """
    client_params_tensor = torch.cat([param.flatten() for name, param in client_params.items() 
                                    if "weight" in name or "bias" in name])
    global_params_tensor = torch.cat([param.flatten() for name, param in global_params.items() 
                                    if "weight" in name or "bias" in name])
    
    return F.cosine_similarity(client_params_tensor, global_params_tensor, dim=0)

def cos_sim_layer(client_params: Dict[str, torch.Tensor], global_params: Dict[str, torch.Tensor]):
    """ 
    Calculate the cosine similarity between client and global parameters layer by layer.
    Args:
        client_params: the client parameters (Dict[str, torch.Tensor])
        global_params: the global parameters (Dict[str, torch.Tensor])
    Returns:
        sim: the cosine similarity between the client and global parameters (torch.tensor)
    """
    sim = torch.tensor(0.0, device="cuda")
    sim_count = 0
    
    for name, client_param in client_params.items():
        if "weight" in name or "bias" in name:
            global_param = global_params[name]
            sim += F.cosine_similarity(client_param.flatten(), global_param.flatten(), dim=0)
            sim_count += 1

    return sim / sim_count

def model_dist(client_params: Dict[str, torch.Tensor], global_params: Dict[str, torch.Tensor]):
    """
    Calculate the l2 distance between client and global parameters.
    Args:
        client_params: the client parameters (Dict[str, torch.Tensor])
        global_params: the global parameters (Dict[str, torch.Tensor])
    Returns:
        dist: the distance between the client and global parameters (torch.tensor)
    """
    client_params_tensor = torch.cat([param.flatten() for name, param in client_params.items() 
                                    if "weight" in name or "bias" in name])
    global_params_tensor = torch.cat([param.flatten() for name, param in global_params.items() 
                                    if "weight" in name or "bias" in name])
    
    return torch.norm(client_params_tensor - global_params_tensor, p=2)

def model_dist_layer(client_params: Dict[str, torch.Tensor], global_params: Dict[str, torch.Tensor]):
    """
    Calculate the l2 distance between client and global parameters layer by layer.
    Args:
        client_params: the client parameters (Dict[str, torch.Tensor])
        global_params: the global parameters (Dict[str, torch.Tensor])
    Returns:
        dist: the distance between the client and global parameters (torch.tensor)
    """
    dist = torch.tensor(0.0, device="cuda")
    
    for name, client_param in client_params.items():
        if "weight" in name or "bias" in name:
            global_param = global_params[name]
            dist += torch.norm((client_param - global_param).flatten(), p=2)

    return dist
