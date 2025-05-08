"""
Implementation of FLTrust server for federated learning.
"""

import torch
import torch.nn.functional as F

from typing import Dict, List, Tuple
from logging import INFO
from fl_bdbench.servers.defense_categories import RobustAggregationServer
from fl_bdbench.utils.logging_utils import log

class FLTrustServer(RobustAggregationServer):
    """
    FLTrust server implementation that uses cosine similarity with trusted data
    to assign trust scores to client updates.
    """

    def __init__(self, server_config, server_type = "fltrust", eta: float = 1.0):
        super().__init__(server_config, server_type, eta)
        self.central_update = None

    def _parameters_dict_to_vector(self, net_dict: Dict) -> torch.Tensor:
        """Convert parameters dictionary to flat vector, excluding batch norm parameters."""
        vec = []
        for key, param in net_dict.items():
            if any(x in key.split('.')[-1] for x in ['num_batches_tracked', 'running_mean', 'running_var']):
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLTrust mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        if self.central_update is None:
            log(INFO, "FLTrust: No central update available, using standard FedAvg")
            return super().aggregate_client_updates(client_updates)

        # Convert central update to vector
        central_vector = self._parameters_dict_to_vector(self.central_update)
        central_norm = torch.linalg.norm(central_vector)

        score_list = []
        total_score = 0
        sum_parameters = {}

        for _, _, local_update in client_updates:
            # Convert local update to vector
            local_vector = self._parameters_dict_to_vector(local_update)

            # Calculate cosine similarity and trust score
            client_cos = F.cosine_similarity(central_vector, local_vector, dim=0)
            client_cos = max(client_cos.item(), 0)
            client_norm_ratio = central_norm / torch.linalg.norm(local_vector)

            score_list.append(client_cos)
            total_score += client_cos

            # Accumulate weighted updates
            for key, param in local_update.items():
                if key not in sum_parameters:
                    sum_parameters[key] = client_cos * client_norm_ratio * param.clone().to(self.device)
                else:
                    sum_parameters[key].add_(client_cos * client_norm_ratio * param.to(self.device))

        log(INFO, f"FLTrust scores: {score_list}")

        # If all scores are 0, return current global model
        if total_score == 0:
            log(INFO, "FLTrust: All trust scores are 0, keeping current model")
            return False

        # Update global model parameters in-place
        for key, param in self.global_model_params.items():
            if key.endswith('num_batches_tracked'):
                continue
            else:
                update = (sum_parameters[key] / total_score)
                param.add_(update * self.eta)

        return True

    def set_central_update(self, central_update: Dict):
        """Set the trusted central update for scoring."""
        self.central_update = central_update
