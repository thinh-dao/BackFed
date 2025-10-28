"""
Trimmed Mean server implementation for FL.
"""
import torch

from logging import INFO
from typing import List, Tuple
from backfed.servers.base_server import BaseServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples

class TrimmedMeanServer(BaseServer):
    """
    Server that implements trimmed mean aggregation to mitigate the impact of malicious clients.

    Trimmed mean removes a specified percentage of the largest and smallest values before
    computing the mean, making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_config, server_type="trimmed_mean", eta=1.0, trim_ratio=0.2):
        """
        Initialize the trimmed mean server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            trim_ratio: Percentage of clients to trim from each end (default: 0.2)
        """
        super(TrimmedMeanServer, self).__init__(server_config, server_type)
        self.eta = eta
        self.trim_ratio = trim_ratio
        log(INFO, f"Initialized Trimmed Mean server with trim_ratio={trim_ratio}")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using trimmed mean.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client parameters
        updates = [update for _, _, update in client_updates]
        num_clients = len(client_updates)

        # Calculate number of clients to trim from each end
        num_trim = int(num_clients * self.trim_ratio)

        # Accumulate gradient updates with coordinate-wise trimmed mean 
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device, dtype=torch.float32)
            for name, param in self.global_model.state_dict().items()
        }    

        # Coordinate-wise trimmed mean over trainable params
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            # We only perform trimmed-mean on trainable params
            if name not in self.trainable_names:
                for update in updates:
                    param_update = update[name].to(device=self.device, dtype=torch.float32)
                    weight_accumulator[name].add_(param_update * 1/num_clients)
            else:
                # Stack the selected updates for this layer
                layer_updates = torch.stack([
                    update[name].to(device=self.device, dtype=torch.float32)
                    for update in updates
                ])

                # Sort along client dimension
                sorted_updates, _ = torch.sort(layer_updates, dim=0)

                # Calculate trimmed mean
                trimmed_updates = sorted_updates[num_trim:num_clients-num_trim]
                mean_update = torch.mean(trimmed_updates, dim=0).to(param.device)

                # Update weight_accumulator
                weight_accumulator[name].copy_(mean_update.to(param.device))

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

    def __repr__(self) -> str:
        return f"TrimmedMean(trim_ratio={self.trim_ratio})"
