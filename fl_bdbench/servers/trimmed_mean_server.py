"""
Trimmed Mean server implementation for FL.
"""
import torch
from typing import List, Tuple

from fl_bdbench.servers.defense_categories import RobustAggregationServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict
from logging import INFO

class TrimmedMeanServer(RobustAggregationServer):
    """
    Server that implements trimmed mean aggregation to mitigate the impact of malicious clients.

    Trimmed mean removes a specified percentage of the largest and smallest values before
    computing the mean, making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_config, server_type="trimmed_mean", trim_ratio=0.2):
        """
        Initialize the trimmed mean server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            trim_ratio: Percentage of clients to trim from each end (default: 0.2)
        """
        super(TrimmedMeanServer, self).__init__(server_config, server_type)
        self.trim_ratio = trim_ratio
        log(INFO, f"Initialized Trimmed Mean server with trim_ratio={trim_ratio}")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> bool:
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
        client_params = [params for _, _, params in client_updates]
        num_clients = len(client_params)

        # Calculate number of clients to trim from each end
        num_trim = int(num_clients * self.trim_ratio)

        # Update global model parameters directly
        for name, param in self.global_model_params.items():
            if name.endswith('num_batches_tracked'):
                continue

            # Stack parameters from all clients for this layer
            layer_updates = torch.stack([client_param[name] for client_param in client_params])

            # Sort values along the client dimension
            sorted_updates, _ = torch.sort(layer_updates, dim=0)

            # Calculate trimmed mean
            trimmed_updates = sorted_updates[num_trim:num_clients-num_trim]
            mean_update = torch.mean(trimmed_updates, dim=0)

            # Apply update
            param.copy_(mean_update.to(param.device))

        return True

    def __repr__(self) -> str:
        return f"TrimmedMean(trim_ratio={self.trim_ratio})"
