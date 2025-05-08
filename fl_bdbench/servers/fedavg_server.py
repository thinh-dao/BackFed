"""
FedAvg server implementation for FL.
"""
import torch

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.utils.logging_utils import log
from logging import INFO, WARNING
from typing import List, Tuple
from fl_bdbench.const import StateDict

class UnweightedFedAvgServer(BaseServer):
    """
    FedAvg server with equal client weights, following 'How To Backdoor Federated Learning'.

    Formula: G^{t+1} = G^{t} + eta/m * sum_{i=1}^{m} (L_i^{t+1} - G^{t})
    where G^t: global model, m: num clients, L_i: client model, eta: learning rate
    """

    def __init__(self, server_config, server_type = "unweighted_fedavg", eta=0.1, **kwargs):
        super(UnweightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized UnweightedFedAvg server with eta={eta}")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]):
        """
        Aggregate client updates using FedAvg with equal weights.
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False

        num_clients = len(client_updates)

        # Cumulative model updates with equal weights
        weight = 1 / num_clients
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model_params.items()
        }

        for _, _, client_state in client_updates:
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

class WeightedFedAvgServer(BaseServer):
    """
    FedAvg server with client weights proportional to their number of samples.
    """

    def __init__(self, server_config, server_type="weighted_fedavg", eta=1.0, **kwargs):
        super(WeightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized Weighted FedAvg server with eta={eta}")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]):
        """
        Aggregate client updates using FedAvg with weights proportional to number of samples.
        """
        if not client_updates:
            return False

        # Cumulative model updates with weights proportional to number of samples
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model_params.items()
        }

        total_samples = sum(num_samples for _, num_samples, _ in client_updates)
        for _, num_samples, client_state in client_updates:
            weight = (num_samples / total_samples)
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
