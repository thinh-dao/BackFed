"""
Geometric Median server implementation for FL.
"""
import torch

from typing import List, Tuple
from logging import INFO
from backfed.servers.fedavg_server import BaseServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples

class CoordinateMedianServer(BaseServer):
    """
    Server that implements coordinate-wise median aggregation to mitigate the impact of malicious clients.

    Coordinate-wise median computes the median for each parameter independently,
    making it robust against extreme values from malicious clients.
    """

    def __init__(self, server_config, server_type="coordinate_median", eta=1.0):
        """
        Initialize the coordinate-wise median server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
        """
        super(CoordinateMedianServer, self).__init__(server_config, server_type)
        self.eta = eta
        log(INFO, f"Initialized Coordinate-wise Median server")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using coordinate-wise median.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client model updates
        updates = [update for _, _, update in client_updates]
        num_clients = len(client_updates)
        
        # Accumulate gradient updates with coordinate-wise median
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device, dtype=torch.float32)
            for name, param in self.global_model.state_dict().items()
        }    

        # Coordinate-wise median over trainable params
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            # We only perform trimmed-mean on trainable params
            if name not in self.trainable_names:
                for update in updates:
                    param_update = update[name].to(device=self.device, dtype=torch.float32)
                    weight_accumulator[name].add_(param_update * 1/num_clients)
            else:
                # Stack parameters from all clients for this layer
                layer_updates = torch.stack([
                    client_model_update[name].to(device=self.device, dtype=param.dtype)
                    for client_model_update in updates
                ])

                # Update weight_accumulator
                weight_accumulator[name].copy_(torch.median(layer_updates, dim=0).values)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

class GeometricMedianServer(BaseServer):
    """
    Server that implements geometric median aggregation to mitigate the impact of malicious clients.

    Geometric median finds the point that minimizes the sum of distances to all client updates,
    making it robust against Byzantine attacks. Uses flattened parameter vectors for efficient computation.
    """

    def __init__(self, server_config, server_type="geometric_median", eta=1.0, eps=1e-5, maxiter=4, ftol=1e-6):
        """
        Initialize the geometric median server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            eta: Learning rate for global model update
            eps: Smallest allowed value of denominator to avoid divide by zero
            maxiter: Maximum number of Weiszfeld iterations
            ftol: Tolerance for function value convergence
        """
        super(GeometricMedianServer, self).__init__(server_config, server_type)
        self.eta = eta
        self.eps = eps
        self.maxiter = maxiter
        self.ftol = ftol
        log(INFO, f"Initialized GeometricMedian server with eta={eta}, eps={eps}, maxiter={maxiter}, ftol={ftol}")

    def _l2distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Calculate L2 distance between two flattened parameter vectors."""
        return torch.linalg.norm(p1 - p2)

    def _geometric_median_objective(self, median: torch.Tensor, points: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Compute the weighted sum of distances from median to all points."""
        distances = torch.stack([self._l2distance(p, median) for p in points])
        return torch.sum(distances * weights) / torch.sum(weights)

    def _weighted_average(self, points: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted average of flattened parameter vectors."""
        weights = weights / weights.sum()
        result = points[0] * weights[0]
        for i in range(1, len(points)):
            result += points[i] * weights[i]
        return result

    @torch.no_grad()
    def _geometric_median(self, points: List[torch.Tensor], weights: torch.Tensor = None, eps: float = None, maxiter: int = None, ftol: float = None) -> torch.Tensor:
        """
        Compute geometric median using Weiszfeld algorithm.

        Args:
            points: List of flattened parameter vectors
            weights: Tensor of weights for each point (defaults to uniform weights)
            eps: Smoothing parameter to avoid division by zero
            maxiter: Maximum number of iterations
            ftol: Tolerance for function value convergence

        Returns:
            Geometric median of the points (flattened vector)
        """
        # Use default values from instance if not provided
        if eps is None:
            eps = self.eps
        if maxiter is None:
            maxiter = self.maxiter
        if ftol is None:
            ftol = self.ftol

        # Store original alphas (client weights) - these stay constant
        if weights is None:
            alphas = torch.ones(len(points), device=self.device)
        else:
            alphas = weights.clone()
        alphas = alphas / alphas.sum()

        # Initialize median estimate at weighted mean
        median = self._weighted_average(points, alphas)
        objective_value = self._geometric_median_objective(median, points, alphas)

        log(INFO, f"Initial objective value: {objective_value.item()}")

        # Weiszfeld iterations
        for iteration in range(maxiter):
            prev_obj_value = objective_value
            
            # Compute distances and iteration-specific weights
            distances = torch.stack([self._l2distance(p, median) for p in points])
            iter_weights = alphas / torch.clamp(distances, min=eps)
            iter_weights = iter_weights / iter_weights.sum()
            median = self._weighted_average(points, iter_weights)

            # Objective uses original alphas
            objective_value = self._geometric_median_objective(median, points, alphas)
            log(INFO, f"GeometricMedian Iteration {iteration}: Objective value: {objective_value.item()}")
            
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                log(INFO, f"GeometricMedian converged after {iteration + 1} iterations")
                break

        return median

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using geometric median on flattened parameter vectors.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client updates
        updates = [update for _, _, update in client_updates]
        num_clients = len(client_updates)
        
        # Convert trainable parameters to flattened vectors for geometric median
        update_vectors = []
        for update in updates:
            update_vector = self.parameters_dict_to_vector(update)
            update_vectors.append(update_vector)
        
        # Equal weights for all clients
        weights = torch.ones(len(update_vectors), device=self.device)
        
        # Compute geometric median for trainable parameters
        geometric_median_vector = self._geometric_median(
            update_vectors, 
            weights, 
            eps=self.eps, 
            maxiter=self.maxiter, 
            ftol=self.ftol
        )
        
        # Apply trainable parameters from geometric median vector
        offset = 0
        for name, param in self.global_model.named_parameters():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            numel = param.numel()
            delta = geometric_median_vector[offset : offset + numel].view_as(param)
            param.data.add_(delta.to(param.device) * self.eta)
            offset += numel

        # Handle batch normalization parameters (running_mean, running_var) - average from all clients
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            if "running_mean" in name or "running_var" in name:
                # Aggregate from all clients using simple average
                bn_update = sum(update[name].to(self.device, dtype=torch.float32) 
                                for update in updates) / num_clients
                param.data.add_(bn_update * self.eta)
        
        return True

    def __repr__(self) -> str:
        return f"GeometricMedian(eta={self.eta}, eps={self.eps}, maxiter={self.maxiter}, ftol={self.ftol})"
