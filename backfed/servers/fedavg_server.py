"""
FedAvg server implementation for FL.
"""
import torch

from backfed.servers.base_server import BaseServer
from backfed.utils.logging_utils import log
from logging import INFO, WARNING
from typing import List, Tuple
from backfed.const import ModelUpdate, client_id, num_examples

class FedAvgServer(BaseServer):
    def weight_accumulator(self, updates: ModelUpdate, weights: List[float]) -> ModelUpdate:
        """
        Accumulate weighted client updates with corresponding weights.

        Args:
            updates: List of model updates (\Delta_w)
            weights: Corresponding weights for update
        Returns:
            weight_accumulator: Accumulated weighted updates
        """
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device, dtype=torch.float32)
            for name, param in self.global_model.state_dict().items()
        }

        for weight, update in zip(weights, updates):
            for name, param_update in update.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue

                param_update = param_update.to(device=self.device, dtype=torch.float32)
                weight_accumulator[name].add_(param_update * weight)

        return weight_accumulator

class UnweightedFedAvgServer(FedAvgServer):
    """
    FedAvg server with equal client weights, following standard FedAvg algorithm.

    Formula: G^{t+1} = (1/m) * sum_{i=1}^{m} L_i^{t+1}
    where G^t: global model, m: num clients, L_i: client model
    """

    def __init__(self, server_config, server_type = "unweighted_fedavg", eta=1.0, **kwargs):
        super(UnweightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized UnweightedFedAvg server with eta={eta}")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]):
        """
        Aggregate client updates using FedAvg with equal weights.
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False

        num_clients = len(client_updates)

        # Report client-global model distances
        for client_id_val, _, client_update in client_updates:
            distance = self.compute_client_distance(client_update)
            log(INFO, f"Client {client_id_val} has weight diff norm {distance:.4f}")

        # Cumulative model updates with equal weights
        weights = [1 / num_clients] * num_clients
        updates = [model_update for _, _, model_update in client_updates]
        weight_accumulator = self.weight_accumulator(updates, weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

class WeightedFedAvgServer(FedAvgServer):
    """
    FedAvg server with client weights proportional to their number of samples.
    """

    def __init__(self, server_config, server_type="weighted_fedavg", eta=1.0, **kwargs):
        super(WeightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized Weighted FedAvg server with eta={eta}")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]):
        """
        Aggregate client updates using FedAvg with weights proportional to number of samples.
        """
        if not client_updates:
            return False

        # Report client-global model distances
        for client_id_val, _, client_update in client_updates:
            distance = self.compute_client_distance(client_update)
            log(INFO, f"Client {client_id_val} has weight diff norm {distance:.4f}")

        # Most Pythonic - unpack in one go
        updates, num_samples_list = [], []
        for _, n_samples, update in client_updates:
            num_samples_list.append(n_samples)
            updates.append(update)

        total_samples = sum(num_samples_list)
        weights = [n / total_samples for n in num_samples_list]
        weight_accumulator = self.weight_accumulator(updates, weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True
