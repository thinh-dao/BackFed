"""
Implementation of Robust Learning Rate Server for federated learning.
This defense adjusts learning rates based on sign agreement among client updates.
"""

import torch

from typing import List, Tuple
from logging import INFO
from fl_bdbench.servers.defense_categories import RobustAggregationServer
from fl_bdbench.const import StateDict, client_id, num_examples
from fl_bdbench.utils import log

class RobustLRServer(RobustAggregationServer):
    """
    RobustLR server implementation that adjusts learning rates based on sign agreement
    among client updates to defend against backdoor attacks.
    """

    def __init__(self, server_config, server_type="robustlr",
                 robustLR_threshold: float = 0.0,
                 eta: float = 0.1):
        """
        Initialize RobustLR server.

        Args:
            server_config: Server configuration
            server_type: Type of server
            robustLR_threshold: Threshold for sign agreement to determine learning rate
                               (0.0 means no robust learning rate adjustment)
            eta: Server learning rate
        """
        super().__init__(server_config, server_type)
        self.robustLR_threshold = robustLR_threshold
        self.eta = eta
        log(INFO, f"Initialized RobustLR server with threshold={robustLR_threshold}, eta={eta}")

    def _parameters_dict_to_vector(self, state_dict: StateDict) -> torch.Tensor:
        """Convert parameters dictionary to flat vector."""
        vec = []
        for name, param in state_dict.items():
            if 'bias' in name or 'weight' in name:
                vec.append(param.view(-1))
        return torch.cat(vec).to(self.device)

    def _compute_robustLR(self, client_updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute robust learning rates based on sign agreement.

        Args:
            client_updates: List of client update vectors
        Returns:
            lr_vector: Vector of learning rates for each parameter
        """
        # Get sign of each update
        client_updates_sign = [torch.sign(update) for update in client_updates]

        # Sum the signs and take absolute value to measure agreement
        sum_of_signs = torch.abs(sum(client_updates_sign))

        # Create learning rate vector based on threshold
        lr_vector = torch.ones_like(sum_of_signs) * self.eta
        lr_vector[sum_of_signs < self.robustLR_threshold] = -self.eta

        return lr_vector

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
        """
        Aggregate client updates using RobustLR mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract updates and convert to vectors
        update_vectors = []
        weights = []

        global_vector = self._parameters_dict_to_vector(self.global_model_params)
        for client_id, num_examples, update in client_updates:
            # Convert update to vector
            update_vector = self._parameters_dict_to_vector(update)

            # Calculate difference from global model
            diff_vector = update_vector - global_vector

            update_vectors.append(diff_vector)
            weights.append(num_examples)

        # Compute robust learning rates
        lr_vector = self._compute_robustLR(update_vectors)

        # Compute weighted average of updates
        total_weight = sum(weights)
        weighted_updates = torch.zeros_like(update_vectors[0])

        for w, update in zip(weights, update_vectors):
            weighted_updates += (w / total_weight) * update

        # Apply learning rates to updates
        weighted_updates *= lr_vector

        # Update global model parameters
        global_vector = self._parameters_dict_to_vector(self.global_model_params)
        new_global_vector = global_vector + weighted_updates

        # Convert vector back to state dict
        idx = 0
        for name, param in self.global_model_params.items():
            if 'bias' in name or 'weight' in name:
                param_size = param.numel()
                param_shape = param.shape

                # Extract the corresponding segment from the new global vector
                param_vector = new_global_vector[idx:idx+param_size]
                self.global_model_params[name] = param_vector.reshape(param_shape)

                idx += param_size

        log(INFO, f"RobustLR: Applied learning rates with {(lr_vector > 0).sum().item()} positive and {(lr_vector < 0).sum().item()} negative rates")

        return True
