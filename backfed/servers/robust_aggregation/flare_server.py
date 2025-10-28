"""Implementation of FLARE server for federated learning."""

import torch
import copy

from backfed.models.supcon import SupConModel
from backfed.servers.fedavg_server import FedAvgServer
from typing import Dict, List, Tuple
from logging import INFO
from backfed.utils.logging_utils import log

class FlareServer(FedAvgServer):
    """
    FLARE server implementation that uses Maximum Mean Discrepancy (MMD)
    to detect and filter malicious updates.

    This is a hybrid defense that combines anomaly detection (MMD-based detection)
    with robust aggregation (weighted aggregation based on trust scores).
    """

    def __init__(
        self,
        server_config,
        server_type: str = "flare",
        eta: float = 0.1,
        m: int = 10, # Number of auxiliary data samples
        aux_class: int = 5, # The class used as auxiliary data
    ):  
        self.m = m
        self.aux_class = aux_class
        self.eta = eta
        
        super().__init__(server_config, server_type) # Setup datasets and so on
        
        log(
            INFO,
            f"Initialized FLARE server with m={self.m}, aux_class={self.aux_class}, eta={self.eta}",
        )
        
    def _prepare_dataset(self):
        """We override the _prepare_dataset function to load auxiliary clean data for the defense."""
        super()._prepare_dataset()
                                    
        # Sample m indices of the auxiliary class from the training set
        chosen_indices = []
        targets = getattr(self.testset, 'targets', None) or getattr(self.testset, 'labels', None)
        
        if targets is not None:
            for idx, label in enumerate(targets):
                if label == self.aux_class:
                    chosen_indices.append(idx)
                    if len(chosen_indices) >= self.m:
                        break
        else:
            # Fallback: iterate through dataset to find samples of aux_class
            for idx in range(len(self.testset)):
                if len(chosen_indices) >= self.m:
                    break
                sample = self.testset[idx]
                label = sample[1] if isinstance(sample, (tuple, list)) and len(sample) > 1 else None
                if label == self.aux_class:
                    chosen_indices.append(idx)

        if len(chosen_indices) < self.m:
            raise ValueError(f"Flare: Not enough samples of class {self.aux_class} in the test set.")

        aux_tensors = torch.stack([self.testset[i][0] for i in chosen_indices])
        if self.normalization:
            self.aux_inputs = self.normalization(aux_tensors).to(self.device)
        else:
            self.aux_inputs = aux_tensors.to(self.device)
            
    def _kernel_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between two sets of vectors."""
        sigma = 1.0
        return torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2 * sigma**2))

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy between two sets of features."""
        m, n = x.size(0), y.size(0)

        if m == 0 or n == 0:
            return torch.tensor(0.0, device=x.device if m else y.device)

        xx_kernel = self._kernel_function(x, x)
        yy_kernel = self._kernel_function(y, y)
        xy_kernel = self._kernel_function(x, y)

        if m > 1:
            xx_sum = (xx_kernel.sum() - torch.diagonal(xx_kernel).sum()) / (m * (m - 1))
        else:
            xx_sum = torch.tensor(0.0, device=xx_kernel.device)

        if n > 1:
            yy_sum = (yy_kernel.sum() - torch.diagonal(yy_kernel).sum()) / (n * (n - 1))
        else:
            yy_sum = torch.tensor(0.0, device=yy_kernel.device)

        xy_sum = xy_kernel.sum() / (m * n)

        return xx_sum + yy_sum - 2 * xy_sum

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLARE mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        client_features = []
        client_ids = []
        updates = []

        for client_id, _, model_update in client_updates:
            client_ids.append(client_id)
            updates.append(model_update)

            # Load client model update into a temporary model
            temp_model = copy.deepcopy(self.global_model)
            for name, param in temp_model.state_dict().items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                param.data.add_(model_update[name] * self.eta)
            
            # Get feature_extractor
            temp_model.eval()
            feature_extractor = SupConModel(temp_model)
            feature_extractor.to(self.device).eval()

            # Add client features
            with torch.no_grad():
                features = feature_extractor(self.aux_inputs)
            
            client_features.append(features)
        
        num_clients = len(client_updates)
        distance_matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                mmd_score = self._compute_mmd(client_features[i], client_features[j]).item()
                distance_matrix[i, j] = distance_matrix[j, i] = mmd_score
        
        # log(INFO, "FLARE distances: %s", distance_matrix.tolist())

        neighbor_count = int(0.5 * num_clients)
        vote_counter = torch.zeros(num_clients, dtype=torch.float32)

        for i in range(num_clients):
            distances = distance_matrix[i]
            sorted_indices = torch.argsort(distances)
            neighbor_indices = [idx.item() for idx in sorted_indices if idx != i][:neighbor_count]
            for neighbor in neighbor_indices:
                vote_counter[neighbor] += 1

        flare_weights = torch.softmax(vote_counter, dim=0).tolist()
        log(INFO, f"FLARE (client_id, weight): {list(zip(client_ids, flare_weights))}")

        weight_accumulator = self.weight_accumulator(updates, flare_weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True
