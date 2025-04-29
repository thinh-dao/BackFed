"""
Implementation of DeepSight defense against backdoor attacks in FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import copy
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import hdbscan
from functools import reduce
from logging import INFO, WARNING

from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import IMG_SIZE, NUM_CLASSES

class DeepSightServer(UnweightedFedAvgServer):
    """
    DeepSight: A defense mechanism against backdoor attacks in Federated Learning.
    Uses clustering-based approach to detect and filter malicious updates.
    """

    def __init__(self, 
                 server_config,
                 num_seeds: int = 3, 
                 num_samples: int = 20000, 
                 deepsight_batch_size: int = 1000, 
                 deepsight_tau: float = 1.0/3,
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
        super(DeepSightServer, self).__init__(server_config, **kwargs)
        self.num_seeds = num_seeds
        self.num_samples = num_samples
        self.deepsight_batch_size = deepsight_batch_size
        self.deepsight_tau = deepsight_tau
        
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]):
        if len(client_updates) == 0:
            return 

        # Extract local model states
        local_model_updates = [params for _, _, params in client_updates]
        
        # Calculate metrics for detection
        TEs, NEUPs, ed = [], [], []
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        
        # Calculate update norms and NEUPs
        for local_model_update in local_model_updates:
            # Calculate Euclidean distance
            squared_sum = sum(
                np.sum(np.power(param - self.global_model.state_dict()[name], 2))
                for name, param in local_model_update.items()
                if "bias" in name or "weight" in name
            )
            ed.append(math.sqrt(squared_sum))

            # Calculate NEUPs for last layer
            last_layer_params = [(k, v) for k, v in local_model_update.items() if "classifier" in k or "fc" in k]
            if len(last_layer_params) >= 2:
                diff_weight = last_layer_params[-2][1] - self.global_model.state_dict()[last_layer_params[-2][0]]
                diff_bias = last_layer_params[-1][1] - self.global_model.state_dict()[last_layer_params[-1][0]]
                
                UPs = np.abs(diff_bias) + np.sum(np.abs(diff_weight), axis=1)
                NEUP = UPs ** 2 / np.sum(UPs ** 2)
                NEUPs.append(NEUP)
                TE = sum(1 for j in NEUP if j >= (1 / num_classes) * np.max(NEUP))
                TEs.append(TE)

        log(INFO, "DeepSight: Finished calculating metrics")
        
        # Label clients based on TE threshold
        labels = [0 if te >= np.median(TEs) / 2 else 1 for te in TEs]
        
        # Calculate different distance metrics
        DDifs = self._calculate_ddifs(local_model_updates)
        cosine_distances = self._calculate_cosine_distances(local_model_updates)
        
        # Perform clustering
        cosine_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(cosine_distances)
        cosine_cluster_dists = self._dists_from_clust(cosine_clusters, len(local_model_updates))
        
        NEUPs = np.reshape(NEUPs, (len(local_model_updates), num_classes))
        neup_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(NEUPs)
        neup_cluster_dists = self._dists_from_clust(neup_clusters, len(local_model_updates))
        
        # Process DDif clusters
        ddif_cluster_dists = []
        for i in range(self.num_seeds):
            DDifs[i] = np.reshape(DDifs[i], (len(local_model_updates), num_classes))
            ddif_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(DDifs[i])
            ddif_cluster_dists.append(self._dists_from_clust(ddif_clusters, len(local_model_updates)))
        
        # Merge distances and perform final clustering
        merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        merged_distances = np.mean([
            merged_ddif_cluster_dists,
            neup_cluster_dists,
            cosine_cluster_dists
        ], axis=0)
        
        final_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        
        # Identify benign clients
        benign_clients = []
        labels = np.asarray(labels)
        for cluster in np.unique(final_clusters):
            if cluster == -1:
                indexes = np.argwhere(final_clusters == cluster).flatten()
                benign_clients.extend([i for i in indexes if labels[i] == 1])
            else:
                indexes = np.argwhere(final_clusters == cluster).flatten()
                if np.sum(labels[indexes]) / len(indexes) < self.deepsight_tau:
                    benign_clients.extend(indexes)
        
        log(INFO, f"DeepSight: Selected {len(benign_clients)} benign clients")
        
        # Aggregate updates from benign clients
        if not benign_clients:
            log(WARNING, "No benign clients found, using global model")
            return self.global_model.state_dict()
        
        benign_updates = [client_updates[i] for i in benign_clients]
        return super().aggregate_client_updates(benign_updates)

    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Calculate distance matrix from cluster assignments."""
        return np.array([[0 if clusters[i] == clusters[j] else 1 for j in range(N)] for i in range(N)])
        
    def _calculate_cosine_distances(self, local_model_updates: List[Dict]) -> np.ndarray:
        """Calculate cosine distances between client updates."""
        N = len(local_model_updates)
        distances = np.zeros((N, N))
        
        def _cos_sim(x: np.ndarray, y: np.ndarray) -> float:
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)
        
        # Get last layer parameters
        last_layer_params = [(k, v) for k, v in local_model_updates[0].items() if "classifier" in k or "fc" in k]
        last_layer_name = last_layer_params[-1][0]
        
        for i in range(N):
            for j in range(i + 1, N):
                update_i = local_model_updates[i][last_layer_name] - self.global_model.state_dict()[last_layer_name]
                update_j = local_model_updates[j][last_layer_name] - self.global_model.state_dict()[last_layer_name]
                
                dist = 1.0 - _cos_sim(update_i.flatten(), update_j.flatten())
                distances[i, j] = distances[j, i] = dist
                
        return distances

    def _calculate_ddifs(self, local_model_updates: List[Dict]) -> np.ndarray:
        """Calculate DDifs using random noise inputs."""
        device = next(self.global_model.parameters()).device
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        img_size = IMG_SIZE[self.config.dataset.upper()]
        
        DDifs = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset(img_size, self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)
            
            for local_update in local_model_updates:
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(local_update)
                local_model.eval()
                
                DDif = torch.zeros(num_classes, device=device)
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        output_local = torch.softmax(local_model(inputs), dim=1)
                        output_global = torch.softmax(self.global_model(inputs), dim=1)
                        
                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))
                
                DDif /= self.num_samples
                DDifs.append(DDif.cpu().numpy())
                
        return np.array(DDifs).reshape(self.num_seeds, len(local_model_updates), -1)

class NoiseDataset(torch.utils.data.Dataset):
    """Dataset that generates random noise inputs."""
    
    def __init__(self, size: Tuple[int, int, int], num_samples: int):
        self.size = size
        self.num_samples = num_samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        noise = torch.rand(self.size)
        return noise, 0