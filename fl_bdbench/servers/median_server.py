"""
Geometric Median server implementation for FL.
"""
import torch
import numpy as np

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict
from logging import INFO
from typing import List, Tuple
from scipy.optimize import minimize

class CoordinateMedianServer(BaseServer):
    """
    Server that implements coordinate-wise median aggregation to mitigate the impact of malicious clients.

    Coordinate-wise median computes the median for each parameter independently,
    making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_lr, server_type="coordinate_median"):
        """
        Initialize the coordinate-wise median server.

        Args:
            server_lr: Dictionary containing configuration
            server_type: Type of server
        """
        super(CoordinateMedianServer, self).__init__(server_lr, server_type)
        log(INFO, f"Initialized Coordinate-wise Median server")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> bool:
        """
        Aggregate client updates using coordinate-wise median.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client parameters
        client_params = [params for _, _, params in client_updates]

        # Process each layer separately
        for name, param in self.global_model_params.items():
            if name.endswith('num_batches_tracked'):
                continue
            # Stack parameters from all clients for this layer
            layer_params = torch.stack([client_param[name] for client_param in client_params])
            # Update global model parameters directly with median
            param.copy_(torch.median(layer_params, dim=0).values)

        return True


class GeometricMedianServer(BaseServer):
    """
    Server that implements geometric median aggregation to mitigate the impact of malicious clients.

    Geometric median finds the point that minimizes the sum of distances to all client updates,
    making it robust against Byzantine attacks.
    """

    def __init__(self, config, server_type="geometric_median"):
        """
        Initialize the geometric median server.

        Args:
            config: Dictionary containing configuration
            server_type: Type of server
        """
        super(GeometricMedianServer, self).__init__(config, server_type)
        log(INFO, f"Initialized Geometric Median server")

    def _geometric_median_objective(self, median_candidate: np.ndarray, points: np.ndarray) -> float:
        """Compute the sum of distances from median candidate to all points."""
        return sum(np.linalg.norm(p - median_candidate) for p in points)

    def _compute_geometric_median(self, client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Compute the geometric median of client weights.
        
        Args:
            client_weights: List of client model weights
            
        Returns:
            Geometric median of client weights
        """
        # Flatten and concatenate weights for each client
        flattened_weights = []
        shapes = []
        for weights in client_weights:
            flat_list = []
            shapes_list = []
            for w in weights:
                shapes_list.append(w.shape)
                flat_list.append(w.flatten())
            flattened = np.concatenate(flat_list)
            flattened_weights.append(flattened)
            shapes = shapes_list
        
        # Convert to numpy array
        flattened_weights = np.array(flattened_weights)
        
        # Use mean as initial guess
        initial_guess = np.mean(flattened_weights, axis=0)
        
        # Compute geometric median
        result = minimize(
            self._geometric_median_objective,
            initial_guess,
            args=(flattened_weights,),
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        geometric_median = result.x
        
        # Reshape back to original shapes
        aggregated_weights = []
        start_idx = 0
        for shape in shapes:
            size = np.prod(shape)
            layer_weights = geometric_median[start_idx:start_idx+size].reshape(shape)
            aggregated_weights.append(layer_weights)
            start_idx += size
        
        return aggregated_weights

    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> bool:
        """
        Aggregate client updates using geometric median.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client parameters
        client_params = [params for _, _, params in client_updates]

        # Convert client parameters to list of numpy arrays for geometric median
        numpy_params = []
        for params in client_params:
            numpy_layer_params = []
            for name in params.keys():
                numpy_layer_params.append(params[name].cpu().numpy())
            numpy_params.append(numpy_layer_params)

        # Compute geometric median
        aggregated_weights = self._compute_geometric_median(numpy_params)

        # Update global model parameters directly
        for i, (name, param) in enumerate(self.global_model_params.items()):
            if name.endswith('num_batches_tracked'):
                continue
            param.copy_(torch.tensor(
                aggregated_weights[i],
                device=param.device,
                dtype=param.dtype
            ))

        return True

