"""
Implementation of DeepSight defense against backdoor attacks in FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import copy
import math
import numpy as np
import torch
import hdbscan

from typing import List, Tuple
from logging import INFO
from fl_bdbench.servers.defense_categories import AnomalyDetectionServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import IMG_SIZE, NUM_CLASSES, StateDict, client_id, num_examples

class DeepSightServer(AnomalyDetectionServer):
    """
    DeepSight: A defense mechanism against backdoor attacks in Federated Learning.
    Uses clustering-based approach to detect and filter malicious updates.
    """

    def __init__(self,
                 server_config,
                 num_seeds: int = 3,
                 num_samples: int = 20000,
                 deepsight_batch_size: int = 2000,
                 deepsight_tau: float = 1.0/3,
                 server_type: str = "deepsight",
                 **kwargs) -> None:
        """
        Initialize DeepSight server.

        Args:
            server_config: Server configuration
            num_seeds: Number of random seeds for DDif calculation
            num_samples: Number of noise samples
            deepsight_batch_size: Batch size for DDif calculation
            deepsight_tau: Threshold for determining benign clusters
        """
        super(DeepSightServer, self).__init__(server_config, server_type, **kwargs)
        self.num_seeds = num_seeds
        self.num_samples = num_samples
        self.deepsight_batch_size = deepsight_batch_size
        self.deepsight_tau = deepsight_tau
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies in the updates using DeepSight.

        Args:
            client_updates: List of client updates

        Returns:
            List of indices of client_updates classified as anomalous
        """
        # Extract local model states
        local_model_updates = []
        client_ids = []
        for client_id, _, params in client_updates:
            # Move params to the same device as global model
            params_on_device = {name: param.to(self.device) for name, param in params.items()}
            local_model_updates.append(params_on_device)
            client_ids.append(client_id)

        # Calculate metrics for detection
        TEs, NEUPs, ed = [], [], []
        num_classes = NUM_CLASSES[self.config.dataset.upper()]

        # Calculate update norms and NEUPs
        for local_model_update in local_model_updates:
            # Calculate Euclidean distance
            squared_sum = 0
            for name, param in local_model_update.items():
                if "bias" in name or "weight" in name:
                    diff = param - self.global_model.state_dict()[name].to(self.device)
                    squared_sum += torch.sum(torch.pow(diff, 2)).item()
            ed.append(math.sqrt(squared_sum))

            # Calculate NEUPs for last layer
            last_layer_params = [(k, v) for k, v in local_model_update.items() if "classifier" in k or "fc" in k]
            if len(last_layer_params) >= 2:
                diff_weight = last_layer_params[-2][1] - self.global_model.state_dict()[last_layer_params[-2][0]].to(self.device)
                diff_bias = last_layer_params[-1][1] - self.global_model.state_dict()[last_layer_params[-1][0]].to(self.device)
                
                # Keep calculations on GPU as long as possible
                abs_diff_bias = torch.abs(diff_bias)
                abs_diff_weight = torch.abs(diff_weight)
                
                # Sum along appropriate dimension on GPU
                sum_abs_weight = torch.sum(abs_diff_weight, dim=1)
                
                # Combine on GPU
                UPs = abs_diff_bias + sum_abs_weight
                
                # Square and normalize on GPU
                UPs_squared = UPs ** 2
                UPs_sum = torch.sum(UPs_squared)
                NEUP = UPs_squared / UPs_sum
                
                # Convert to CPU/numpy to calculate TE
                NEUP_np = NEUP.cpu().numpy()
                NEUPs.append(NEUP_np)
                
                # Calculate TE
                max_NEUP = np.max(NEUP_np)
                threshold = (1 / num_classes) * max_NEUP
                TE = sum(1 for j in NEUP_np if j >= threshold)
                TEs.append(TE)

        log(INFO, "DeepSight: Finished calculating metrics")

        # Label clients based on TE threshold (following reference implementation)
        # Use True for benign (low TE) and False for malicious (high TE)
        classificat_boundary = np.median(TEs)
        labels = [False if te > classificat_boundary * 0.5 else True for te in TEs]

        # Calculate different distance metrics
        DDifs = self._calculate_ddifs(local_model_updates)
        cosine_distances = self._calculate_cosine_distances(local_model_updates)

        # Perform clustering
        # For cosine distances, use precomputed metric
        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cosine_distances)
        cosine_cluster_dists = self._dists_from_clust(cosine_clusters, len(local_model_updates))

        NEUPs = np.reshape(NEUPs, (len(local_model_updates), num_classes))
        neup_clusters = hdbscan.HDBSCAN().fit_predict(NEUPs)
        neup_cluster_dists = self._dists_from_clust(neup_clusters, len(local_model_updates))

        # Process DDif clusters
        ddif_cluster_dists = []
        for i in range(self.num_seeds):
            DDifs[i] = np.reshape(DDifs[i], (len(local_model_updates), num_classes))
            ddif_clusters = hdbscan.HDBSCAN().fit_predict(DDifs[i])
            ddif_cluster_dists.append(self._dists_from_clust(ddif_clusters, len(local_model_updates)))

        # Merge distances and perform final clustering
        merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        merged_distances = np.mean([
            merged_ddif_cluster_dists,
            neup_cluster_dists,
            cosine_cluster_dists
        ], axis=0)

        final_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        log(INFO, f"DeepSight: Final clusters: {np.unique(final_clusters, return_counts=True)}")

        # Identify benign clients (following reference implementation)
        benign_clients_indices = []
        labels = np.asarray(labels)
        
        # Count positives (benign clients) in each cluster
        positive_counts = {}
        total_counts = {}
        for i, cluster in enumerate(final_clusters):
            if cluster != -1:
                if cluster in positive_counts:
                    positive_counts[cluster] += 1 if not labels[i] else 0  # False = malicious
                    total_counts[cluster] += 1
                else:
                    positive_counts[cluster] = 1 if not labels[i] else 0
                    total_counts[cluster] = 1
        
        # Select clients from benign clusters
        for i, cluster in enumerate(final_clusters):
            if cluster != -1:
                # Check if cluster is mostly malicious
                if cluster in positive_counts:
                    malicious_ratio = positive_counts[cluster] / total_counts[cluster]
                    if malicious_ratio < self.deepsight_tau:  # If malicious ratio is low enough
                        benign_clients_indices.append(i)
            else:
                # For noise cluster, only include clients labeled as benign
                if labels[i]:  # True = benign
                    benign_clients_indices.append(i)

        log(INFO, f"DeepSight: Selected {len(benign_clients_indices)} benign clients")
        malicious_clients_indices = [i for i in range(len(client_updates)) if i not in benign_clients_indices]

        benign_clients = [client_ids[i] for i in benign_clients_indices]
        malicious_clients = [client_ids[i] for i in malicious_clients_indices]
        return benign_clients, malicious_clients

    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Calculate distance matrix from cluster assignments (following reference)."""
        pairwise_dists = np.ones((N, N))
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i][j] = 0
        return pairwise_dists

    def _calculate_cosine_distances(self, local_model_updates: List[StateDict]) -> np.ndarray:
        """Calculate cosine distances between client updates."""
        N = len(local_model_updates)
        distances = np.zeros((N, N))

        # Get last layer parameters
        last_layer_params = [(k, v) for k, v in local_model_updates[0].items() if "classifier" in k or "fc" in k]
        bias_name = last_layer_params[-1][0]  # Assuming bias is the last parameter

        for i in range(N):
            for j in range(i + 1, N):
                # Get bias differences
                bias_i = local_model_updates[i][bias_name] - self.global_model.state_dict()[bias_name].to(self.device)
                bias_j = local_model_updates[j][bias_name] - self.global_model.state_dict()[bias_name].to(self.device)
                
                # Calculate cosine distance using PyTorch (preserving your optimization)
                bias_i_flat = bias_i.flatten()
                bias_j_flat = bias_j.flatten()
                
                dot_product = torch.dot(bias_i_flat, bias_j_flat)
                norm_i = torch.linalg.norm(bias_i_flat)
                norm_j = torch.linalg.norm(bias_j_flat)
                
                similarity = dot_product / (norm_i * norm_j + 1e-10)
                dist = 1.0 - similarity.item()
                
                distances[i, j] = distances[j, i] = dist

        return distances

    def _calculate_ddifs(self, local_model_updates: List[StateDict]) -> np.ndarray:
        """Calculate DDifs using random noise inputs."""
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        img_height, img_width, num_channels = IMG_SIZE[self.config.dataset.upper()]

        DDifs = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset((num_channels, img_height, img_width), self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)
            local_model = copy.deepcopy(self.global_model)
            
            seed_ddifs = []
            for local_update in local_model_updates:
                local_model.load_state_dict(local_update)
                local_model.eval()

                DDif = torch.zeros(num_classes, device=self.device)
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        # Apply softmax to outputs before division (following reference)
                        output_local = torch.softmax(local_model(inputs), dim=1)
                        output_global = torch.softmax(self.global_model(inputs), dim=1)

                    # Division and summation (preserving your optimization)
                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))

                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())
            
            DDifs.append(seed_ddifs)

        return np.array(DDifs)

class NoiseDataset(torch.utils.data.Dataset):
    """Dataset that generates random noise inputs."""

    def __init__(self, size: Tuple[int, int, int], num_samples: int):
        self.size = size
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return just the noise tensor without a label
        noise = torch.rand(self.size)
        return noise
