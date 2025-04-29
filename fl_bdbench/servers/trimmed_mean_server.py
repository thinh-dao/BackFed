"""
Trimmed Mean server implementation for FL.
"""
import torch
from typing import List, Tuple

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict
from logging import INFO

def trimmed_mean(
    client_updates: List[Tuple[int, int, StateDict]], 
    global_model_state: StateDict,
    trim_ratio: float = 0.2
) -> StateDict:
    """
    Aggregate client updates using trimmed mean.

    Args:
        client_updates: List of tuples (client_id, num_examples, model_update)
        global_model_state: Current global model state dict
        trim_ratio: Percentage of clients to trim from each end (default: 0.2)
    Returns:
        The global model state dict after aggregation
    """
    if len(client_updates) == 0:
        return global_model_state

    # Extract client parameters
    client_params = [params for _, _, params in client_updates]
    num_clients = len(client_params)

    # Calculate number of clients to trim
    num_trim = int(num_clients * trim_ratio)

    # Create a new state dict for the aggregated model
    aggregated_state_dict = {}

    # Process each layer separately
    for name in global_model_state.keys():
        # Stack parameters from all clients for this layer
        layer_params = torch.stack([client_param[name] for client_param in client_params])

        # Sort values along client dimension
        sorted_values, _ = torch.sort(layer_params, dim=0)

        # Trim the smallest and largest values
        if num_trim > 0:
            trimmed_values = sorted_values[num_trim:-num_trim]
        else:
            trimmed_values = sorted_values

        # Average the remaining values
        aggregated_state_dict[name] = torch.mean(trimmed_values, dim=0)

    return aggregated_state_dict

class TrimmedMeanServer(BaseServer):
    """
    Server that implements trimmed mean aggregation to mitigate the impact of malicious clients.

    Trimmed mean removes a specified percentage of the largest and smallest values before
    computing the mean, making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_lr, server_type="trimmed_mean", trim_ratio=0.2):
        """
        Initialize the trimmed mean server.

        Args:
            server_lr: Dictionary containing configuration
            server_type: Type of server
            trim_ratio: Percentage of clients to trim from each end (default: 0.2)
        """
        super(TrimmedMeanServer, self).__init__(server_lr, server_type)
        self.trim_ratio = trim_ratio
        log(INFO, f"Initialized Trimmed Mean server with trim_ratio={trim_ratio}")

    @torch.no_grad()
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
