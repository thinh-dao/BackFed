"""
RFLBAT (Robust Federated Learning with Backdoor Attack Tolerance) server implementation.
This version uses PCA and clustering-based detection of malicious updates.
"""
import numpy as np
import logging
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import os

from logging import WARNING
from fl_bdbench.utils.logging_utils import log
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.const import StateDict
from typing import List, Tuple

class RFLBATServer(UnweightedFedAvgServer):
    """
    RFLBAT server that uses PCA and clustering to detect and filter malicious updates.
    """
    
    def __init__(self, 
                 server_lr, 
                 server_type="rflbat",
                 eps1=10.0,  # First-stage filtering threshold
                 eps2=4.0,   # Second-stage filtering threshold
                 save_plots=True):
        super(RFLBATServer, self).__init__(server_lr, server_type)
        self.eps1 = eps1
        self.eps2 = eps2
        self.save_plots = save_plots
        
        # Create directory for plots if needed
        if self.save_plots:
            self.plot_dir = os.path.join(server_lr.output_dir, "rflbat_plots")
            os.makedirs(self.plot_dir, exist_ok=True)

    def _flatten_model_updates(self, updates: StateDict) -> np.ndarray:
        """Flatten model updates into a single vector."""
        flattened = []
        for name, param in updates.items():
            if 'fc' in name or 'layer4' in name:  # Focus on important layers
                flattened.extend(param.cpu().numpy().flatten())
        return np.array(flattened)

    def _gap_statistics(self, data: np.ndarray, max_clusters=10, n_refs=5) -> int:
        """
        Compute optimal number of clusters using Gap Statistics.
        """
        gaps = np.zeros(max_clusters)
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++')
            kmeans.fit(data)
            
            # Calculate within-cluster dispersion
            cluster_dispersion = kmeans.inertia_
            
            # Generate reference datasets
            ref_dispersions = []
            for _ in range(n_refs):
                rand_data = np.random.uniform(low=data.min(), high=data.max(), 
                                           size=data.shape)
                kmeans.fit(rand_data)
                ref_dispersions.append(kmeans.inertia_)
            
            # Calculate gap statistic
            gap = np.log(np.mean(ref_dispersions)) - np.log(cluster_dispersion)
            gaps[k-1] = gap
        
        # Find optimal number of clusters
        optimal_k = np.argmax(gaps) + 1
        return optimal_k

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

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> StateDict:
        """
        Aggregate client updates using RFLBAT defensive mechanism.
        """
        if len(client_updates) == 0:
            return self.global_model.state_dict()

        # Extract and flatten updates
        flattened_updates = []
        client_ids = []
        for client_id, _, update in client_updates:
            flattened = self._flatten_model_updates(update)
            flattened_updates.append(flattened)
            client_ids.append(client_id)
        
        X = np.array(flattened_updates)

        # Apply PCA
        pca = PCA(n_components=2)
        X_dr = pca.fit_transform(X)

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
        n_clusters = self._gap_statistics(X_filtered)
        
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
            
            cluster_updates = X[cluster_indices]
            similarities = smp.cosine_similarity(cluster_updates)
            cluster_scores.append(np.median(np.mean(similarities, axis=1)))

        best_cluster = np.argmin(cluster_scores)
        accepted_indices = [accepted_indices[i] for i in range(len(cluster_labels))
                          if cluster_labels[i] == best_cluster]

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

        if self.save_plots:
            self._save_pca_plot(X_dr, final_accepted)
        
        benign_updates = [client_updates[i] for i in final_accepted]
        return super().aggregate_client_updates(benign_updates)
