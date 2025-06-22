"""
WeakDP server implementation for differential privacy with fixed clipping.
"""
import torch
from typing import List, Tuple
from logging import INFO

from backfed.servers.defense_categories import RobustAggregationServer, ClientSideDefenseServer
from backfed.utils.logging_utils import log
from backfed.const import StateDict, client_id, num_examples

class NormClippingServer(RobustAggregationServer):
    """
    Server that clips the norm of client updates to defend against poisoning attacks.
    """
    def __init__(self, server_config, server_type="norm_clipping", clipping_norm=5.0, eta=0.1, verbose=True):
        """
        Args:
            server_type: Type of server.
            clipping_norm: Clipping norm for the norm clipping.
            eta: Learning rate for the server.
        """
        super(NormClippingServer, self).__init__(server_config, server_type)
        self.clipping_norm = clipping_norm
        self.eta = eta
        self.verbose = verbose
        log(INFO, f"Initialized NormClipping server with clipping_norm={clipping_norm}, eta={eta}")

    def clip_updates_inplace(self, client_ids: List[client_id], client_diffs: List[StateDict]) -> None:
        """
        Clip the norm of client_diffs (L_i - G) in-place.

        Args:
            client_diffs: List of client_diffs (state dicts)
        """
        for client_id, client_diff in zip(client_ids, client_diffs):
            flatten_weights = []
            for name, param in client_diff.items():
                if 'weight' in name or 'bias' in name:
                    flatten_weights.append(param.view(-1))

            if not flatten_weights:
                continue
            
            flatten_weights = torch.cat(flatten_weights)
            weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2)

            if self.verbose:
                log(INFO, f"Client {client_id} has weight diff norm {weight_diff_norm}")

            if weight_diff_norm > self.clipping_norm:
                scaling_factor = self.clipping_norm / weight_diff_norm
                for name, param in client_diff.items():
                    if 'weight' in name or 'bias' in name:
                        client_diff[name].mul_(scaling_factor)

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> StateDict:
        """Aggregate client updates with norm clipping."""
        if len(client_updates) == 0:
            return False

        # Clip updates
        client_diffs = []
        client_weights = []
        client_ids = []
        for client_id, num_examples, client_params in client_updates:
            diff_dict = {}
            for name, param in client_params.items():
                if name.endswith('num_batches_tracked'):
                    continue
                diff_dict[name] = param.to(self.device) - self.global_model_params[name]
            client_diffs.append(diff_dict)
            client_weights.append(num_examples)
            client_ids.append(client_id)

        self.clip_updates_inplace(client_ids, client_diffs)
        client_weights = torch.tensor(client_weights, device=self.device)
        client_weights = client_weights / client_weights.sum()

        # Update global model with clipped weight differences
        for i, client_diff in enumerate(client_diffs):
            for name, diff in client_diff.items():
                if name.endswith('num_batches_tracked'):
                    continue
                self.global_model_params[name].add_(diff * client_weights[i] * self.eta)

        return True

class WeakDPServer(ClientSideDefenseServer, NormClippingServer):
    """
    Server that implements differential privacy with fixed clipping and Gaussian noise.
    """
    def __init__(self, server_config, server_type="weakdp", strategy="unweighted_fedavg",
                 std_dev=0.025, clipping_norm=5.0):

        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            strategy: Strategy for the server.
            std_dev: Standard deviation for the Gaussian noise.
            clipping_norm: Clipping norm for the Gaussian noise.
        """
        super(WeakDPServer, self).__init__(server_config, server_type)

        if std_dev < 0:
            raise ValueError("The std_dev should be a non-negative value.")
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        self.std_dev = std_dev
        self.clipping_norm = clipping_norm
        self.strategy = strategy
        log(INFO, f"Initialized WeakDP server with std_dev={std_dev}, clipping_norm={clipping_norm}")

    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> StateDict:
        """Aggregate client updates with DP guarantees."""
        super().aggregate_client_updates(client_updates)
        self.add_gaussian_noise_inplace(self.global_model_params)
        return True

    def add_gaussian_noise_inplace(self, state_dict: StateDict) -> None:
        """Add Gaussian noise to model parameters."""
        for name, param in state_dict.items():
            if 'weight' in name or 'bias' in name:
                noise = torch.normal(0, self.std_dev, param.shape, device=param.device)
                param.add_(noise)

    def __repr__(self) -> str:
        return f"WeakDP(strategy={self.strategy}, std_dev={self.std_dev}, clipping_norm={self.clipping_norm})"