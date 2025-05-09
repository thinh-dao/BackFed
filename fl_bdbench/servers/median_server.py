"""
Geometric Median server implementation for FL.
"""
import torch
import numpy as np

from fl_bdbench.servers.defense_categories import RobustAggregationServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict, client_id, num_examples
from logging import INFO
from typing import List, Tuple
from scipy.optimize import minimize

class CoordinateMedianServer(RobustAggregationServer):
    """
    Server that implements coordinate-wise median aggregation to mitigate the impact of malicious clients.

    Coordinate-wise median computes the median for each parameter independently,
    making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_config, server_type="coordinate_median"):
        """
        Initialize the coordinate-wise median server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
        """
        super(CoordinateMedianServer, self).__init__(server_config, server_type)
        log(INFO, f"Initialized Coordinate-wise Median server")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
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


class GeometricMedianServer(RobustAggregationServer):
    """
    Server that implements geometric median aggregation to mitigate the impact of malicious clients.

    Geometric median finds the point that minimizes the sum of distances to all client updates,
    making it robust against Byzantine attacks.
    """

    def __init__(self, server_config, server_type="geometric_median", eps=1e-5, maxiter=4, ftol=1e-6):
        """
        Initialize the geometric median server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            eps: Smallest allowed value of denominator to avoid divide by zero
            maxiter: Maximum number of Weiszfeld iterations
            ftol: Tolerance for function value convergence
        """
        super(GeometricMedianServer, self).__init__(server_config, server_type)
        self.eps = eps
        self.maxiter = maxiter
        self.ftol = ftol
        log(INFO, f"Initialized Geometric Median server with eps={eps}, maxiter={maxiter}, ftol={ftol}")

    @torch.no_grad()
    def _l2distance(self, p1, p2):
        """Calculate L2 distance between two lists of tensors."""
        return torch.linalg.norm(torch.stack([torch.linalg.norm(x1 - x2) for (x1, x2) in zip(p1, p2)]))

    @torch.no_grad()
    def _geometric_median_objective(self, median, points, weights):
        """Compute the weighted sum of distances from median to all points."""
        distances = torch.tensor([self._l2distance(p, median).item() for p in points], device=self.device)
        return torch.sum(distances * weights) / torch.sum(weights)

    def _weighted_average_component(self, points, weights):
        """Compute weighted average for a single component."""
        ret = points[0] * weights[0]
        for i in range(1, len(points)):
            ret += points[i] * weights[i]
        return ret

    def _weighted_average(self, points, weights):
        """Compute weighted average across all components."""
        weights = weights / weights.sum()
        return [self._weighted_average_component(component, weights=weights) for component in zip(*points)]

    def _geometric_median(self, points, weights, eps=1e-6, maxiter=4, ftol=1e-6):
        """
        Compute geometric median using Weiszfeld algorithm.

        Args:
            points: List of points, where each point is a list of tensors
            weights: Tensor of weights for each point
            eps: Smoothing parameter to avoid division by zero
            maxiter: Maximum number of iterations
            ftol: Tolerance for function value convergence

        Returns:
            SimpleNamespace with median estimate and convergence information
        """
        with torch.no_grad():
            # Initialize median estimate at weighted mean
            median = self._weighted_average(points, weights)
            new_weights = weights
            objective_value = self._geometric_median_objective(median, points, weights)

            log(INFO, f"Initial objective value: {objective_value.item()}")

            # Weiszfeld iterations
            for iter in range(maxiter):
                prev_obj_value = objective_value
                denom = torch.stack([self._l2distance(p, median) for p in points])
                new_weights = weights / torch.clamp(denom, min=eps) 
                median = self._weighted_average(points, new_weights)

                objective_value = self._geometric_median_objective(median, points, weights)
                log(INFO, f"Iteration {iter}: Objective value: {objective_value.item()}")
                if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                    break
            
        median = self._weighted_average(points, new_weights)  # for autodiff
        return median

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
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
        
        # Convert client parameters to list of tensors for geometric median
        points = []
        for params in client_params:
            point = []
            for name, param in self.global_model_params.items():
                if not name.endswith('num_batches_tracked'):
                    point.append(params[name].to(self.device))
            points.append(point)
        
        # Equal weights for all clients
        weights = torch.ones(len(points), device=self.device)
        
        # Compute geometric median
        geometric_median = self._geometric_median(
            points, 
            weights, 
            eps=self.eps, 
            maxiter=self.maxiter, 
            ftol=self.ftol
        )
        
        # Update global model parameters directly
        i = 0
        for name, param in self.global_model_params.items():
            if not name.endswith('num_batches_tracked'):
                param.copy_(geometric_median[i])
                i += 1

        log(INFO, f"Geometric median aggregation completed")
        return True
