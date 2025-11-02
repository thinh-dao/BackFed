"""
Multi-Krum server implementation for FL.

This implements the Multi-Krum algorithm from the paper:
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
by Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer.
"""
import torch

from .anomaly_detection_server import AnomalyDetectionServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples
from logging import INFO, WARNING
from typing import List, Tuple

class ADMultiKrumServer(AnomalyDetectionServer):
    """
    Version of Multi-Krum that overrides AnomalyDetectionServer to keep track of detection metrics.
    Basic functionality remains the same. We explicitly assume the number of malicious clients each round is known.

    Multi-Krum selects a subset of client updates that are closest to each other,
    making it robust against Byzantine attacks where a fraction of clients may be malicious.

    The algorithm works by:
    1. Computing pairwise distances between client updates
    2. For each client, finding the closest n-f-2 clients (where n is num_clients_per_round and f is number of malicious clients)
    3. Summing these distances to get a score for each client
    4. Selecting the k clients with the lowest scores (where k is the number of clients to keep)
    5. Aggregating the updates from these selected clients
    """

    def __init__(self, server_config, server_type="ad_multi_krum", selection_ratio=0.5, oracle=True, eta=0.1):
        """
        Initialize the Multi-Krum server.

        Args:
            eta: Learning rate for applying the aggregated updates
        """
        super(ADMultiKrumServer, self).__init__(server_config, server_type, eta)

        self.oracle = oracle

        if self.oracle:
            self.num_malicious_clients = None
            log(INFO, f"Initialized ADMulti-Krum server with known number of malicious clients each round, eta={self.eta}")
        else:
            self.num_clients_to_keep = int(self.config.num_clients_per_round * selection_ratio)  
            log(INFO, f"Initialized ADMulti-Krum server with num_clients_to_keep={self.num_clients_to_keep}, eta={self.eta}")      

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]], **kwargs) -> Tuple[int, int]:
        """
        Detect anomalies in the client updates using Multi-Krum algorithm.
        """
        if len(client_updates) == 0:
            return False
        
        if self.oracle:
            # Assuming the number of malicious clients each round is known
            num_malicious_clients = len(self.client_manager.malicious_clients_per_round[self.current_round])
            num_clients_to_keep = self.config.num_clients_per_round - num_malicious_clients
        else:
            num_malicious_clients = self.config.num_clients_per_round - self.num_clients_to_keep
            num_clients_to_keep = self.num_clients_to_keep

        # Extract client parameters
        client_ids = [client_id for client_id, _, _ in client_updates]
        flattened_params = [self.parameters_dict_to_vector(params).cpu() for _, _, params in client_updates]
        num_clients = len(flattened_params)

        # If we have fewer clients than the number to keep, use all clients
        if num_clients <= num_clients_to_keep:
            log(INFO, f"Number of clients ({num_clients}) is less than or equal to num_clients_to_keep "
                     f"({num_clients_to_keep}). Using all clients.")

            return [], client_ids

        # Calculate pairwise squared Euclidean distances
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = torch.linalg.norm(flattened_params[i] - flattened_params[j])**2
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, compute the sum of distances to the closest n-f-2 clients
        n = num_clients
        f = min(num_malicious_clients, n-2)  # Ensure f is valid
        n_neighbors = n - f - 2  # Number of neighbors to consider

        if n_neighbors <= 0:
            log(WARNING, f"Invalid number of neighbors: {n_neighbors}. Using all clients.")
            n_neighbors = n - 1
        
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Get distances to other clients, sort them
            client_distances = distances[i]
            closest_distances, _ = torch.sort(client_distances)
            # Sum the distances to the closest n_neighbors (excluding self, which has distance 0)
            scores[i] = torch.sum(closest_distances[1:n_neighbors+1])

        # Select the clients with the lowest scores
        _, indices = torch.sort(scores)
        selected_indices = indices[:num_clients_to_keep].tolist()

        # Log the selected clients
        benign_client_ids = [client_ids[i] for i in selected_indices]
        malicious_client_ids = [client_id for client_id in client_ids if client_id not in benign_client_ids]
        return malicious_client_ids, benign_client_ids