"""
RFLBAT (Robust Federated Learning with Backdoor Attack Tolerance) server implementation.
This version uses PCA and clustering-based detection of malicious updates.
"""
import numpy as np
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import os
import torch

from logging import WARNING, INFO
from backfed.utils.logging_utils import log
from sklearn.cluster import KMeans
from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.const import StateDict, client_id, num_examples
from typing import List, Tuple

class RFLBATServer(AnomalyDetectionServer):
    """
    RFLBAT server that uses PCA and clustering to detect and filter malicious updates.
    """

    def __init__(self,
                 server_config,
                 server_type="rflbat",
                 eps1=10.0,  # First-stage filtering threshold
                 eps2=4.0,   # Second-stage filtering threshold
                 save_plots=False,
                 eta: float = 0.1):
        
        super(RFLBATServer, self).__init__(server_config, server_type, eta)
        self.eps1 = eps1
        self.eps2 = eps2
        self.save_plots = save_plots

        # Create directory for plots if needed
        if self.save_plots:
            self.plot_dir = os.path.join(server_config.output_dir, "rflbat_plots")
            os.makedirs(self.plot_dir, exist_ok=True)

    def _flatten_model_updates(self, updates: StateDict) -> np.ndarray:
        """Flatten model updates into a single vector."""
        flattened = []
        for name, param in updates.items():
            if 'fc' in name or 'layer4' in name:  # Focus on important layers
                flattened.extend(param.cpu().numpy().flatten())
        return np.array(flattened)

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> Tuple[List[int], List[int]]:
        """Detect anomalies in the client updates."""
        client_ids = []
        update_tensors = []
        
        # First collect all tensors in a list
        for client_id, _, update in client_updates:
            flattened = torch.cat([p.flatten() for p in update.values()], dim=0)
            update_tensors.append(flattened)
            client_ids.append(client_id)
        
        # Stack tensors along a new dimension
        flattened_updates = torch.stack(update_tensors)
        
        # Perform PCA
        U, S, V = torch.pca_lowrank(flattened_updates)
        X_dr = torch.mm(flattened_updates, V[:,:2]).cpu().numpy()

        # First stage filtering based on Euclidean distances
        eu_distances = []
        for i in range(len(X_dr)):
            distances_sum = sum(np.linalg.norm(X_dr[i] - X_dr[j])
                              for j in range(len(X_dr)) if i != j)
            eu_distances.append(distances_sum)

        # First stage acceptance
        median_distance = np.median(eu_distances)
        accepted_indices = [i for i, dist in enumerate(eu_distances)
                          if dist < self.eps1 * median_distance]

        if len(accepted_indices) < 2:
            log(WARNING, "RFLBAT: Too few updates passed first stage filtering. Using standard FedAvg")
            return super().aggregate_client_updates(client_updates)

        # Determine optimal number of clusters
        X_filtered = X_dr[accepted_indices]
        n_clusters = self.gap_statistics(X_filtered, num_sampling=5, \
                                           K_max=9, n=len(X_filtered))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
        cluster_labels = kmeans.fit_predict(X_filtered)

        # Select best cluster based on cosine similarity
        cluster_scores = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) <= 1:
                cluster_scores.append(float('inf'))
                continue

            cluster_updates = flattened_updates[cluster_indices].cpu().numpy()
            similarities = smp.cosine_similarity(cluster_updates)
            cluster_scores.append(np.median(np.average(similarities, axis=1)))

        best_cluster = np.argmin(cluster_scores)
        accepted_indices = [accepted_indices[i] for i in range(len(cluster_labels))
                          if cluster_labels[i] == best_cluster]

        log(INFO, f"RFLBAT First stage: Accepted clients: {[client_ids[i] for i in accepted_indices]}")
        # Second stage filtering
        eu_distances = []
        X_filtered = X_dr[accepted_indices]
        for i in range(len(X_filtered)):
            distances_sum = sum(np.linalg.norm(X_filtered[i] - X_filtered[j])
                              for j in range(len(X_filtered)) if i != j)
            eu_distances.append(distances_sum)

        median_distance = np.median(eu_distances)
        final_accepted = [accepted_indices[i] for i, dist in enumerate(eu_distances)
                         if dist < self.eps2 * median_distance]
        
        log(INFO, f"RFLBAT Second stage: Accepted clients: {[client_ids[i] for i in final_accepted]}")

        if self.save_plots:
            self._save_pca_plot(X_dr, final_accepted)

        benign_clients = [client_ids[i] for i in final_accepted]
        malicious_clients = [client_id for client_id in client_ids if client_id not in benign_clients]
        return malicious_clients, benign_clients

    def _save_pca_plot(self, X_dr: np.ndarray, accepted_indices: List[int]):
        """Save PCA visualization plot."""
        plt.figure(figsize=(10, 8))
        plt.scatter(X_dr[:, 0], X_dr[:, 1], c='gray', alpha=0.5, label='All updates')
        plt.scatter(X_dr[accepted_indices, 0], X_dr[accepted_indices, 1],
                   c='green', label='Accepted updates')
        plt.title(f'PCA visualization - Round {self.current_round}')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f'pca_round_{self.current_round}.png'))
        plt.close()
