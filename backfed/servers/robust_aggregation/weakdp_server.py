"""
WeakDP server implementation for differential privacy with fixed clipping.
"""
import torch

from typing import List, Tuple
from logging import INFO, WARNING
from backfed.servers.fedavg_server import FedAvgServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples

class NormClippingServer(FedAvgServer):
    """
    Server that clips the norm of client updates to defend against poisoning attacks.
    """
    def __init__(self, server_config, server_type="norm_clipping", clipping_norm=5.0, eta=0.1):
        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            clipping_norm: Clipping norm for the norm clipping.
            eta: Learning rate for the server.
        """
        super(NormClippingServer, self).__init__(server_config, server_type)
        self.eta = eta
        self.clipping_norm = clipping_norm
        log(INFO, f"Initialized NormClipping server with clipping_norm={clipping_norm}, eta={eta}")

    @staticmethod
    def scale_update_inplace(update: ModelUpdate, scale_factor: float, clipped_params: List[str]):
        """
        Clip the norm of the update in-place with scale_factor
        Args:
            update: Model update to be clipped (\Delta w)
            scale_factor: Scaling factor to apply
            clipped_params: List of parameter names to be clipped
        """
        for name, param in update.items():
            if name in clipped_params:
                param.data.mul_(scale_factor)

    def clip_updates_inplace(self, client_updates: List[Tuple[client_id, ModelUpdate]]):
        """
        Clip the norm of client_diffs (L_i - G) in-place based on trainable parameters only.

        Args:
            client_updates: List of client updates as tuples of (client_id, num_examples, ModelUpdate)
        """
        for client_id, _, client_diff in client_updates:
            flatten_updates = self.parameters_dict_to_vector(client_diff)
            weight_diff_norm = torch.linalg.norm(flatten_updates, ord=2)

            if weight_diff_norm > self.clipping_norm:
                log(INFO, f"Client {client_id} weight diff norm {weight_diff_norm} -> {self.clipping_norm}")
            
                scaling_factor = self.clipping_norm / weight_diff_norm
                self.scale_update_inplace(client_diff, scaling_factor, self.trainable_names)
            else:
                log(INFO, f"Client {client_id} weight diff norm {weight_diff_norm} within the clipping norm.")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> ModelUpdate:
        """Aggregate client updates with norm clipping."""
        if len(client_updates) == 0:
            log(WARNING, "NormClipping: No client updates found")
            return False

        # Clip client updates    
        self.clip_updates_inplace(client_updates)

        # Cumulative model updates with equal weights
        num_clients = len(client_updates)
        weights = [1 / num_clients] * num_clients
        updates = [model_update for _, _, model_update in client_updates]
        weight_accumulator = self.weight_accumulator(updates, weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

class WeakDPServer(NormClippingServer):
    """
    Server that implements differential privacy with fixed clipping and Gaussian noise.
    """
    def __init__(self, server_config, server_type="weakdp", strategy="weakdp",
                 std_dev=0.025, clipping_norm=5.0, eta=0.1):

        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            strategy: Strategy for the server.
            std_dev: Standard deviation for the Gaussian noise.
            clipping_norm: Clipping norm for the Gaussian noise.
        """
        super(WeakDPServer, self).__init__(server_config, server_type, clipping_norm=clipping_norm, eta=eta)

        if std_dev < 0:
            raise ValueError("The std_dev should be a non-negative value.")
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        self.std_dev = std_dev
        self.strategy = strategy
        log(INFO, f"Initialized WeakDP server with std_dev={std_dev}, clipping_norm={clipping_norm}")

    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> ModelUpdate:
        """Aggregate client updates with DP guarantees by adding Gaussian noise to trainable parameters."""
        if len(client_updates) == 0:
            log(WARNING, "WeakDP: No client updates found")
            return False

        # Clip client updates    
        self.clip_updates_inplace(client_updates)
        
        # Cumulative model updates with equal weights
        num_clients = len(client_updates)
        weights = [1 / num_clients] * num_clients
        updates = [model_update for _, _, model_update in client_updates]
        weight_accumulator = self.weight_accumulator(updates, weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            # Add noise to trainable params
            if name in self.trainable_names:
                noise = torch.normal(0, self.std_dev, param.shape, device=param.device)
                param.data.add_((weight_accumulator[name]+noise) * self.eta)
            else:
                param.data.add_(weight_accumulator[name] * self.eta)
        return True

    def __repr__(self) -> str:
        return f"WeakDP(strategy={self.strategy}, std_dev={self.std_dev}, clipping_norm={self.clipping_norm})"
