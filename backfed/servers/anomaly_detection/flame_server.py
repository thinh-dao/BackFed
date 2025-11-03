"""
Flame server implementation for FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import torch
import numpy as np
import hdbscan

from .anomaly_detection_server import AnomalyDetectionServer
from backfed.servers.robust_aggregation.weakdp_server import NormClippingServer
from logging import INFO, WARNING
from backfed.utils import log, get_last_layer_name
from backfed.const import ModelUpdate, client_id, num_examples
from typing import List, Tuple

class FlameServer(AnomalyDetectionServer):
    """
    Flame server that uses clustering and noise addition to defend against backdoor attacks.

    This is a hybrid defense that combines anomaly detection (clustering) with
    robust aggregation (clipping and noise addition).
    """

    def __init__(self, server_config, server_type="flame", eta=0.5, lamda=0.001):
        super(FlameServer, self).__init__(server_config, server_type, eta)
        self.lamda = lamda
        log(INFO, f"Initialized Flame server with lamda={self.lamda}")
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> Tuple[List[int], List[int], List[float]]:
        """Detect anomalies using HDBSCAN."""
        # Extract client weights and compute distances
        all_update_tensors = []
        euclidean_distances = []
        client_ids = []

        for client_id, _, client_update in client_updates:
            client_ids.append(client_id)
            all_update_tensors.append(self.parameters_dict_to_vector(client_update).cpu().numpy())

            # Calculate euclidean distance
            client_distance = self.compute_client_distance(client_update)
            euclidean_distances.append(client_distance)

        # Cluster clients based on last layer weights
        num_clients = len(client_updates)
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients//2+1,
            min_samples=1,
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(np.array(all_update_tensors, dtype=np.float64))

        # Identify benign clients (largest cluster)
        benign_indices = []
        if labels.max() < 0:
            # No clusters found - treat all as benign
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster = unique_labels[np.argmax(counts)]
            benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster]

        if len(benign_indices) == 0:
            log(WARNING, "Flame: No benign clients found. Treating all as benign.")
            benign_indices = list(range(num_clients))
        
        malicious_clients = [client_ids[idx] for idx in range(len(client_ids)) if idx not in benign_indices]
        benign_clients = [client_ids[idx] for idx in benign_indices]
        return malicious_clients, benign_clients, euclidean_distances

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]):
        """Aggregate client updates using Flame defensive mechanism."""
        if len(client_updates) == 0:
            log(WARNING, "Flame: No client updates found.")
            return False

        # Evaluate detection and log metrics
        malicious_clients, benign_clients, euclidean_distances = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(benign_clients, malicious_clients, true_malicious_clients, len(client_updates))

        # Calculate clip norm from all client distances
        clip_norm = torch.median(torch.tensor(euclidean_distances)).item()

        # Create mapping from client_id to euclidean distance for correct indexing
        client_distance_map = {client_id: euclidean_distances[idx] for idx, (client_id, _, _) in enumerate(client_updates)}

        # Clip benign updates
        for client_id, _, update in client_updates:
            if client_id in malicious_clients:
                continue
            
            client_distance = client_distance_map[client_id]
            if client_distance > clip_norm:
                NormClippingServer.scale_update_inplace(
                    update,
                    scale_factor=min(1.0, clip_norm / client_distance),
                    clipped_params=self.trainable_names
                )
        
        # Aggregate benign updates
        num_clients = len(benign_clients)
        weights = [1 / num_clients] * num_clients
        updates = [update for client_id, _, update in client_updates if client_id in benign_clients]
        weight_accumulator = self.weight_accumulator(updates, weights)

        # Update global model and add noise (following FLAME algorithm lines 13-14)
        # σ = λ · S_t where S_t is the adaptive clipping bound (clip_norm)
        sigma = self.lamda * clip_norm
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

            # Add adaptive noise: G_t* = G_t + N(0, σ²)
            if name in self.trainable_names:
                noise = torch.normal(0, sigma, param.shape, device=param.device)
                param.data.add_(noise)

        return True
