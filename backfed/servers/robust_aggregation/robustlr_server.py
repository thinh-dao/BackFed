"""
Implementation of Robust Learning Rate Server for federated learning.
This defense adjusts learning rates based on sign agreement among client updates.
"""

import torch

from typing import List, Tuple
from logging import INFO, WARNING
from backfed.servers.fedavg_server import FedAvgServer
from backfed.const import ModelUpdate, client_id, num_examples
from backfed.utils import log

class RobustLRServer(FedAvgServer):
    """
    RobustLR server implementation that adjusts learning rates based on sign agreement
    among client updates to defend against backdoor attacks.
    """

    def __init__(self, server_config, server_type="robustlr", eta: float = 0.1, 
                 robustLR_threshold: float = 4):
        """
        Initialize RobustLR server.

        Args:
            server_config: Server configuration
            server_type: Type of server
            robustLR_threshold: Threshold for sign agreement to determine learning rate.
                               Parameters with sign agreement below this threshold get negative LR,
                               effectively reversing their updates. Higher values mean stricter agreement required.
            eta: Server learning rate
        """
        super().__init__(server_config, server_type)
        self.eta = eta
        self.robustLR_threshold = robustLR_threshold
        log(INFO, f"Initialized RobustLR server with threshold={robustLR_threshold}, eta={eta}")

    def _compute_robustLR(self, updates: List[ModelUpdate]) -> dict:
        """
        Compute robust learning rates based on sign agreement for each parameter.

        Args:
            updates: List of model updates as dictionaries
        Returns:
            lr_dict: Dictionary of learning rate tensors for each parameter
        """
        lr_dict = {}
        
        for name in updates[0].keys():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            if name not in self.trainable_names:
                lr_dict[name] = torch.ones_like(sum_of_signs) * self.eta
                continue
            
            # Get signs of all client updates for this parameter
            signs = [torch.sign(update[name].to(self.device)) for update in updates]
            
            # Sum the signs and take absolute value to measure agreement
            sum_of_signs = torch.abs(sum(signs))
            
            # Create learning rate tensor based on threshold
            lr_tensor = torch.ones_like(sum_of_signs) * self.eta
            lr_tensor[sum_of_signs < self.robustLR_threshold] = -self.eta
            
            lr_dict[name] = lr_tensor
        
        return lr_dict

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using RobustLR mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            log(WARNING, "RobustLR: No client updates found")
            return False

        num_clients = len(client_updates)

        # Report client-global model distances
        for client_id_val, _, client_update in client_updates:
            distance = self.compute_client_distance(client_update)
            log(INFO, f"Client {client_id_val} has weight diff norm {distance:.4f}")

        # Extract updates (client_state already contains delta weights)
        updates = [model_update for _, _, model_update in client_updates]

        # Compute robust learning rates for each parameter
        lr_dict = self._compute_robustLR(updates)

        # Compute unweighted average of updates using weight_accumulator
        weights = [1 / num_clients] * num_clients
        averaged_updates = self.weight_accumulator(updates, weights)

        # Apply robust learning rates and update global model
        positive_count = 0
        negative_count = 0
        
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            
            # Apply element-wise robust learning rates
            robust_update = averaged_updates[name] * lr_dict[name]
            param.data.add_(robust_update)
            
            # Count positive and negative learning rates for logging
            positive_count += (lr_dict[name] > 0).sum().item()
            negative_count += (lr_dict[name] < 0).sum().item()

        log(INFO, f"RobustLR: Applied learning rates with {positive_count} positive and {negative_count} negative rates")

        return True
