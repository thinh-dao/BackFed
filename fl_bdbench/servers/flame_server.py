"""
Flame server implementation for FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import torch
import numpy as np
import hdbscan

from logging import INFO, WARNING
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict
from typing import List, Tuple

class FlameServer(UnweightedFedAvgServer):
    """
    Flame server that uses clustering and noise addition to defend against backdoor attacks.
    """
    
    def __init__(self, server_config, server_type="flame", lamda=0.001):
        super(FlameServer, self).__init__(server_config, server_type)
        self.lamda = lamda
        log(INFO, f"Initialized Flame server with lamda={lamda}")

    def _get_last_layers(self, state_dict: StateDict) -> List[str]:
        """Get names of last two layers."""
        layer_names = list(state_dict.keys())
        return layer_names[-2:]
    
    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]):
        """Aggregate client updates using Flame defensive mechanism."""
        if len(client_updates) == 0:
            return False

        # Keep everything on CPU for this function
        last_layers = self._get_last_layers(self.global_model_params)

        # Extract weights and compute distances
        all_client_weights = []
        euclidean_distances = []
        
        for _, _, update in client_updates:
            flat_update = []
            current_client_weight = []
            
            for name, param in update.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.cpu() - self.global_model_params[name]
                    flat_update.append(diff.flatten())  # Keep as torch.Tensor
                
                if name in last_layers:
                    current_client_weight.append(param.cpu().flatten())  # Keep as torch.Tensor
            
            all_client_weights.append(torch.cat(current_client_weight).cpu().numpy()) # Convert to numpy array for HDBSCAN
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update)))  

        # Cluster clients
        num_clients = len(client_updates)
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients//2+1,
            min_samples=1,
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(np.array(all_client_weights))

        # Identify benign clients
        benign_indices = []
        if labels.max() < 0:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster = unique_labels[np.argmax(counts)]
            benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster]

        if len(benign_indices) == 0:
            log(WARNING, "Flame: No benign clients found.")
            return False

        # Aggregate clipped differences from benign clients
        clip_norm = torch.median(torch.stack(euclidean_distances))

        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model_params.items()
        }
        
        for idx in benign_indices:
            _, _, update = client_updates[idx]
            weight = 1 / len(benign_indices)
            for name, param in update.items():
                diff = (param.to(self.device) - self.global_model_params[name])
                if ('weight' in name or 'bias' in name) and euclidean_distances[idx] > clip_norm:
                    diff *= clip_norm / euclidean_distances[idx]
                    
                weight_accumulator[name].add_(diff * weight)

        # Update global model and add noise
        for name, param in self.global_model_params.items():
            if name.endswith('num_batches_tracked'):
                continue
            param.add_(weight_accumulator[name] * self.eta)

            # Add noise to parameters that are not buffer parameters
            if 'weight' in name or 'bias' in name:
                std = self.lamda * clip_norm * torch.std(param)
                noise = torch.normal(0, std, param.shape, device=param.device)
                param.add_(noise)

        return True
