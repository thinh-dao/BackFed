"""
Implementation of FLARE server for federated learning.
"""

import copy
import torch
import numpy as np
from typing import Dict, List, Tuple
from logging import INFO

from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.utils.logging_utils import log

class FlareServer(UnweightedFedAvgServer):
    """
    FLARE server implementation that uses Maximum Mean Discrepancy (MMD)
    to detect and filter malicious updates.
    """
    
    def __init__(self, server_config, voting_threshold: float = 0.5):
        super().__init__(server_config)
        self.voting_threshold = voting_threshold
        self.central_dataset = None
        log(INFO, f"Initialized FLARE server with voting_threshold={voting_threshold}")
        
    def _kernel_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two vectors."""
        sigma = 1.0
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
        
    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy between two sets of features."""
        m, n = x.size(0), y.size(0)
        
        # Compute kernel matrices
        xx_kernel = torch.zeros((m, m))
        yy_kernel = torch.zeros((n, n))
        xy_kernel = torch.zeros((m, n))
        
        for i in range(m):
            for j in range(i, m):
                xx_kernel[i, j] = xx_kernel[j, i] = self._kernel_function(x[i], x[j])
                
        for i in range(n):
            for j in range(i, n):
                yy_kernel[i, j] = yy_kernel[j, i] = self._kernel_function(y[i], y[j])
                
        for i in range(m):
            for j in range(n):
                xy_kernel[i, j] = self._kernel_function(x[i], y[j])
                
        # Calculate MMD statistic
        mmd = (xx_kernel.sum() / (m * (m - 1))) + \
              (yy_kernel.sum() / (n * (n - 1))) - \
              (2 * xy_kernel.sum() / (m * n))
              
        return mmd
        
    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]], 
                               client_features: List[torch.Tensor]) -> bool:
        """
        Aggregate client updates using FLARE mechanism.
        
        Args:
            client_updates: List of (client_id, num_examples, model_update)
            client_features: List of feature representations from clients
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False
            
        if not client_features:
            log(INFO, "FLARE: No client features available, using standard FedAvg")
            return super().aggregate_client_updates(client_updates)
            
        num_clients = len(client_updates)
        
        # Calculate pairwise MMD distances
        distance_list = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                mmd_score = self._compute_mmd(client_features[i], client_features[j])
                distance_list[i].append(mmd_score.item())
                distance_list[j].append(mmd_score.item())
                
        log(INFO, f"FLARE distances: {distance_list}")
        
        # Voting mechanism
        vote_counter = [0] * num_clients
        k = round(num_clients * self.voting_threshold)
        
        for i in range(num_clients):
            sorted_indices = np.argsort(distance_list[i])
            for j in range(k):
                client_id = sorted_indices[j] + 1 if sorted_indices[j] >= i else sorted_indices[j]
                vote_counter[client_id] += 1
                
        # Calculate trust scores
        total_votes = sum(vote_counter)
        if total_votes == 0:
            log(INFO, "FLARE: All trust scores are 0, keeping current model")
            return False
            
        trust_scores = [count/total_votes for count in vote_counter]
        log(INFO, f"FLARE trust scores: {trust_scores}")
        
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model_params.items()
        }

        for weight, (cid, num_samples, client_state) in zip(trust_scores, client_updates):
            for name, param in client_state.items():
                if name.endswith('num_batches_tracked'):
                    continue
                diff = param.to(self.device) - self.global_model_params[name]
                weight_accumulator[name].add_(diff * weight)

        # Update global model with learning rate
        for name, param in self.global_model_params.items():
            if name.endswith('num_batches_tracked'):
                continue
            param.add_(weight_accumulator[name] * self.eta)
            
        return True
