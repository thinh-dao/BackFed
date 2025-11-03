"""
RFLBAT (Robust Federated Learning with Backdoor Attack Tolerance) server implementation.
This version uses PCA and clustering-based detection of malicious updates.
"""
import numpy as np
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import os
import torch

from .anomaly_detection_server import AnomalyDetectionServer
from logging import WARNING, INFO
from backfed.utils.logging_utils import log
from sklearn.cluster import KMeans
from backfed.const import ModelUpdate, client_id, num_examples
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
                 eta: float = 0.5):
        
        super(RFLBATServer, self).__init__(server_config, server_type, eta)
        self.eps1 = eps1
        self.eps2 = eps2
        self.save_plots = save_plots
        self.num_clusters = 2  # Fixed number of clusters for KMeans

        # Create directory for plots if needed
        if self.save_plots:
            self.plot_dir = os.path.join(server_config.output_dir, "rflbat_plots")
            os.makedirs(self.plot_dir, exist_ok=True)

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> Tuple[List[int], List[int]]:
        """Detect anomalies in the client updates."""
        client_ids = []
        update_tensors = []
        
        # First collect all tensors in a list
        for client_id, _, update in client_updates:
            update_tensors.append(self.parameters_dict_to_vector(update))
            client_ids.append(client_id)
        
        # Stack tensors along a new dimension
        flattened_updates = torch.stack(update_tensors)
        
        # Perform PCA
        U, S, V = torch.pca_lowrank(flattened_updates)
        X_dr = torch.mm(flattened_updates, V[:,:2]).cpu().numpy()

        # First stage filtering based on Euclidean distances
        D = smp.euclidean_distances(X_dr)
        eu_distances = D.sum(axis=1)

        # First stage acceptance
        median_distance = np.median(eu_distances)
        accepted_indices = [i for i, dist in enumerate(eu_distances)
                          if dist < self.eps1 * median_distance]

        if len(accepted_indices) < 2:
            log(WARNING, "RFLBAT: Too few updates passed first stage filtering. Using standard FedAvg")
            return super().aggregate_client_updates(client_updates)

        X_filtered = X_dr[accepted_indices]

        # Perform clustering
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++')
        cluster_labels = kmeans.fit_predict(X_filtered)

        # Select best cluster based on cosine similarity
        cluster_scores = []
        for i in range(self.num_clusters):
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
