"""
Multi-Krum server implementation for FL.

This implements the Multi-Krum algorithm from the paper:
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
by Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
"""
import torch
import numpy as np

from fl_bdbench.servers.defense_categories import RobustAggregationServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict
from logging import INFO
from typing import List, Tuple


class MultiKrumServer(RobustAggregationServer):
    """
    Server that implements Multi-Krum aggregation to mitigate the impact of malicious clients.

    Multi-Krum selects a subset of client updates that are closest to each other,
    making it robust against Byzantine attacks where a fraction of clients may be malicious.

    The algorithm works by:
    1. Computing pairwise distances between client updates
    2. For each client, finding the closest m-f-2 clients (where m is total clients and f is number of malicious clients)
    3. Summing these distances to get a score for each client
    4. Selecting the k clients with the lowest scores (where k is the number of clients to keep)
    5. Aggregating the updates from these selected clients
    """

    def __init__(self, server_config, server_type="multi_krum", num_malicious_clients=None, num_clients_to_keep=None, oracle=True, eta=0.1):
        """
        Initialize the Multi-Krum server.

        Args:
            server_config: Dictionary containing configuration
            server_type: Type of server
            num_malicious_clients: Number of malicious clients (f)
            num_clients_to_keep: Number of clients to keep for aggregation (k)
            oracle: If True, use the actual number of malicious clients from the configuration
            eta: Learning rate for applying the aggregated updates
        """
        super(MultiKrumServer, self).__init__(server_config, server_type)

        # If oracle mode is enabled, use the actual number of malicious clients from config
        if oracle:
            self.num_malicious_clients = len(self.config.atk_config.malicious_clients)
        else:
            self.num_malicious_clients = num_malicious_clients if num_malicious_clients is not None else 0

        # Number of clients to keep for aggregation (k)
        self.num_clients_to_keep = num_clients_to_keep if num_clients_to_keep is not None else 1

        # Learning rate for applying the aggregated updates
        self.eta = eta

        log(INFO, f"Initialized Multi-Krum server with num_malicious_clients={self.num_malicious_clients}, "
                 f"num_clients_to_keep={self.num_clients_to_keep}, eta={self.eta}")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> StateDict:
        """
        Aggregate client updates using Multi-Krum algorithm.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            The global model state dict after aggregation
        """
        if len(client_updates) == 0:
            return self.global_model.state_dict()

        # Extract client parameters
        client_ids = [client_id for client_id, _, _ in client_updates]
        client_params = [params for _, _, params in client_updates]
        num_clients = len(client_params)

        # If we have fewer clients than the number to keep, use all clients
        if num_clients <= self.num_clients_to_keep:
            log(INFO, f"Number of clients ({num_clients}) is less than or equal to num_clients_to_keep "
                     f"({self.num_clients_to_keep}). Using all clients.")

            return super().aggregate_client_updates(client_updates)

        # Flatten client parameters for distance calculation
        flattened_params = []
        for params in client_params:
            flat_tensor = torch.cat([p.flatten() for p in params.values()])
            flattened_params.append(flat_tensor.cpu())

        # Calculate pairwise squared Euclidean distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = torch.norm(flattened_params[i] - flattened_params[j])**2
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, compute the sum of distances to the closest m-f-2 clients
        m = num_clients
        f = min(self.num_malicious_clients, m-2)  # Ensure f is valid
        n_neighbors = m - f - 2  # Number of neighbors to consider

        if n_neighbors <= 0:
            log(INFO, f"Invalid number of neighbors: {n_neighbors}. Using all clients.")
            n_neighbors = m - 1

        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Get distances to other clients, sort them
            client_distances = distances[i]
            closest_distances, _ = torch.sort(client_distances)
            # Sum the distances to the closest n_neighbors (excluding self, which has distance 0)
            scores[i] = torch.sum(closest_distances[1:n_neighbors+1])

        # Select the clients with the lowest scores
        _, indices = torch.sort(scores)
        selected_indices = indices[:self.num_clients_to_keep].tolist()

        # Log the selected clients
        selected_client_ids = [client_ids[i] for i in selected_indices]
        log(INFO, f"Multi-Krum selected clients: {selected_client_ids}")

        benign_updates = [client_updates[i] for i in selected_indices]
        return super().aggregate_client_updates(benign_updates)
