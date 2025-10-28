"""
Flame server implementation for FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import torch
import numpy as np
import hdbscan

from .anomaly_detection_server import AnomalyDetectionServer
from logging import INFO, WARNING
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples
from typing import List, Tuple

class FlameServer(AnomalyDetectionServer):
    """
    Flame server that uses clustering and noise addition to defend against backdoor attacks.

    This is a hybrid defense that combines anomaly detection (clustering) with
    robust aggregation (clipping and noise addition).
    """

    def __init__(self, server_config, server_type="flame", lamda=0.001, eta=0.5):
        super(FlameServer, self).__init__(server_config, server_type, eta)
        self.lamda = lamda
        log(INFO, f"Initialized Flame server with lamda={lamda}")

    def _get_last_layers(self, state_dict: ModelUpdate) -> List[str]:
        """Get names of last two layers."""
        layer_names = list(state_dict.keys())
        return layer_names[-2:]
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]):
        # Keep everything on CPU for this function
        client_ids = [client_id for client_id, _, _ in client_updates]
        global_state_dict = {name: param.data for name, param in self.global_model.state_dict().items()}
        last_layers = self._get_last_layers(global_state_dict)

        # Extract weights and compute distances
        all_client_weights = []
        euclidean_distances = []

        for _, _, client_state_dict in client_updates:
            client_weights_tensor = torch.cat([global_state_dict[name].flatten() for name in last_layers])
            all_client_weights.append(client_weights_tensor.cpu().numpy())

            # Calculate norm on GPU for better performance
            client_distance = self.compute_client_distance(client_state_dict)
            euclidean_distances.append(client_distance)

        # Cluster clients
        num_clients = len(client_updates)
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=num_clients//2+1,
            min_samples=1,
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(np.array(all_client_weights, dtype=np.float64))

        # Identify benign clients
        benign_indices = []
        if labels.max() < 0:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster = unique_labels[np.argmax(counts)]
            benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster]

        if len(benign_indices) == 0:
            log(WARNING, "Flame: No benign clients found.")
            return False
        
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
        detection_metrics = self.evaluate_detection(benign_clients, malicious_clients, true_malicious_clients, len(client_updates))

        # Aggregate clipped differences from benign clients
        clip_norm = torch.median(torch.tensor(euclidean_distances))

        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model.state_dict().items()
        }

        weight = 1 / len(benign_clients)
        global_state_dict = self.global_model.state_dict()
        for idx, (client_id, num_examples, update) in enumerate(client_updates):
            if client_id in malicious_clients:
                continue
            
            for name, param in update.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                diff = (param.to(self.device) - global_state_dict[name])
                if euclidean_distances[idx] > clip_norm:
                    diff *= clip_norm / euclidean_distances[idx]
                weight_accumulator[name].add_(diff * weight)

        # Update global model and add noise
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

            # Add noise to parameters that are not buffer parameters
            if "running" not in name and "num_batches_tracked" not in name:
                std = self.lamda * clip_norm.item() * torch.std(param).item()
                noise = torch.normal(0, std, param.shape, device=param.device)
                param.data.add_(noise)

        return True
