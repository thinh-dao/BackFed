"""
Implementation of DeepSight defense against backdoor attacks in FL.
Reference: https://www.usenix.org/conference/raid2020/presentation/fung
"""

import copy
import math
import numpy as np
import torch
import hdbscan

from typing import List, Tuple, Dict
from logging import INFO, WARNING
from backfed.servers.robust_aggregation.weakdp_server import NormClippingServer
from .anomaly_detection_server import AnomalyDetectionServer
from backfed.utils import log, get_last_layer_name
from backfed.const import IMG_SIZE, NUM_CLASSES, ModelUpdate, client_id, num_examples

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
                 deepsight_tau: float = 1/3,
                 server_type: str = "deepsight",
                 eta: float = 0.5,
                 ) -> None:
        """
        Initialize DeepSight server.

        Args:
            server_config: Server configuration
            num_seeds: Number of random seeds for DDif calculation
            num_samples: Number of noise samples
            deepsight_batch_size: Batch size for DDif calculation
            deepsight_tau: Threshold for determining benign clusters
        """
        super(DeepSightServer, self).__init__(server_config, server_type, eta)
        self.num_seeds = num_seeds
        self.num_samples = num_samples
        self.deepsight_batch_size = deepsight_batch_size
        self.deepsight_tau = deepsight_tau
        log(INFO, f"Initialized DeepSight server with deepsight_tau={deepsight_tau}")

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> Tuple[List[int], List[int], List[float]]:
        """
        Detect anomalies in the updates using DeepSight.

        Args:
            client_updates: List of client updates

        Returns:
            malicious_clients, benign_clients, euclidean_distances
        """
        # Extract local model states
        local_model_updates = []
        client_ids = []
        for client_id, _, updates in client_updates:
            local_model_updates.append(updates)
            client_ids.append(client_id)

        # Get last layer name
        last_layer_name = get_last_layer_name(self.global_model)

        # Calculate NEUPs and TEs
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        NEUPs, TEs, euclidean_distances = self._calculate_neups(local_model_updates, num_classes, last_layer_name)

        log(INFO, f"DeepSight: Threshold exceedings: {TEs}")

        # Label clients based on TE threshold (following reference implementation)
        # Use False for benign (high TE) and True for malicious (low TE)
        classification_boundary = np.median(TEs)
        labels = [False if te > classification_boundary * 0.5 else True for te in TEs]

        # Calculate different distance metrics
        DDifs = self._calculate_ddifs(local_model_updates)
        cosine_distances = self._calculate_cosine_distances(local_model_updates, last_layer_name)

        # Perform clustering
        # For cosine distances, use precomputed metric
        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cosine_distances)
        cosine_cluster_dists = self._dists_from_clust(cosine_clusters, len(local_model_updates))

        # Cluster NEUPs
        neup_clusters = hdbscan.HDBSCAN().fit_predict(NEUPs)
        neup_cluster_dists = self._dists_from_clust(neup_clusters, len(local_model_updates))

        # Process DDif clusters
        ddif_cluster_dists = []
        for i in range(self.num_seeds):
            ddif_clusters = hdbscan.HDBSCAN().fit_predict(DDifs[i])
            ddif_cluster_dists.append(self._dists_from_clust(ddif_clusters, len(local_model_updates)))

        # Merge distances and perform final clustering
        merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        merged_distances = np.mean([
            merged_ddif_cluster_dists,
            neup_cluster_dists,
            cosine_cluster_dists
        ], axis=0)

        final_clusters = hdbscan.HDBSCAN().fit_predict(merged_distances)
        log(INFO, f"DeepSight: Final clusters: {final_clusters}")

        # Count positives (benign clients) in each cluster
        positive_counts = {}
        total_counts = {}
        for i, cluster in enumerate(final_clusters):
            if cluster != -1:
                if cluster in positive_counts:
                    positive_counts[cluster] += 1 if not labels[i] else 0
                    total_counts[cluster] += 1
                else:
                    positive_counts[cluster] = 1 if not labels[i] else 0
                    total_counts[cluster] = 1

        # Determine benign and malicious clients
        benign_clients = []
        malicious_clients = []

        # In detect_anomalies (cluster decision):
        for i, cluster in enumerate(final_clusters):
            if cluster != -1:
                amount_of_positives = positive_counts[cluster] / total_counts[cluster]
                if amount_of_positives < self.deepsight_tau:
                    malicious_clients.append(client_ids[i])
                else:
                    benign_clients.append(client_ids[i])
            else:
                # noise cluster
                if labels[i] is False:
                    benign_clients.append(client_ids[i])
                else:
                    malicious_clients.append(client_ids[i])

        log(INFO, f"DeepSight: Selected {len(benign_clients)} benign clients")
        return malicious_clients, benign_clients, euclidean_distances

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, Dict]]):
        """
        AnomalyDetectionServer procedure: Find malicious clients, evaluate detection, and aggregate benign updates.
        If your method performs other operations than just detection (e.g., clipping), you should override this method.

        Args:
            client_updates: List of (client_id, num_examples, model_updates)
        Returns:
            True if the global model parameters are updated, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Detect anomalies & evaluate detection
        malicious_clients, benign_clients, euclidean_distances = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(benign_clients, malicious_clients, true_malicious_clients, len(client_updates))

        # If no benign clients were found, skip update
        if len(benign_clients) == 0:
            log(WARNING, "DeepSight: no benign clients identified, skipping model update.")
            return False
        
        # Aggregate clipped differences from benign clients
        clip_norm = torch.median(torch.tensor(euclidean_distances))

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

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

    def _calculate_neups(self, local_model_updates: List[ModelUpdate], num_classes: int, last_layer_name: str) -> Tuple[List[float], List[float], List[float]]:
        NEUPs, TEs, euclidean_distances = [], [], []

        last_layer_weight_name = last_layer_name + ".weight"
        last_layer_bias_name = last_layer_name + ".bias"

        # Calculate update norms and NEUPs
        for local_model_update in local_model_updates:
            # Calculate Euclidean distance
            client_distance = self.compute_client_distance(local_model_update)
            euclidean_distances.append(client_distance)

            with torch.no_grad():
                # Calculate NEUPs
                diff_weight = torch.sum(torch.abs(local_model_update[last_layer_weight_name]), dim=1) # weight
                diff_bias = torch.abs(local_model_update[last_layer_bias_name]) # bias

                UPs_squared = (diff_bias + diff_weight) ** 2
                NEUP = UPs_squared / torch.sum(UPs_squared)

            NEUP_np = NEUP.cpu().numpy()
            NEUPs.append(NEUP_np)

            # Calculate TE
            max_NEUP = np.max(NEUP_np)
            threshold = max(0.01, 1 / num_classes) * max_NEUP
            TE = sum(1 for j in NEUP_np if j >= threshold)
            TEs.append(TE)

        NEUPs = np.reshape(NEUPs, (len(local_model_updates), num_classes))
        return NEUPs, TEs, euclidean_distances

    def _calculate_ddifs(self, local_model_updates: List[ModelUpdate]) -> np.ndarray:
        """Calculate DDifs using random noise inputs."""
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        img_height, img_width, num_channels = IMG_SIZE[self.config.dataset.upper()]

        self.global_model.eval()
        local_model = copy.deepcopy(self.global_model)
        
        # Cache global state dict and local state dict outside the loops
        global_state_dict = self.global_model.state_dict()
        local_state_dicts = []
        for local_update in local_model_updates:
            state_dict = {name: global_state_dict[name] + local_update[name] 
                                for name in local_update if name not in self.ignore_weights}
            local_state_dicts.append(state_dict)

        DDifs = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset((num_channels, img_height, img_width), self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)
            
            # Pre-generate all noise inputs for this seed
            all_inputs = []
            for inputs in loader:
                all_inputs.append(inputs.to(self.device))
            
            # Compute global model outputs once per seed
            global_outputs = []
            with torch.no_grad():
                for inputs in all_inputs:
                    global_outputs.append(self.global_model(inputs))

            seed_ddifs = []
            for state_dict in local_state_dicts:
                local_model.load_state_dict(state_dict, strict=False)
                local_model.eval()

                DDif = torch.zeros(num_classes, device=self.device)
                with torch.no_grad():
                    for inputs, output_global in zip(all_inputs, global_outputs):
                        output_local = local_model(inputs)

                        # Division and summation
                        ratio = torch.div(output_local, output_global + 1e-30)
                        DDif.add_(ratio.sum(dim=0))

                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())

            DDifs.append(seed_ddifs)

        DDifs = np.reshape(DDifs, (self.num_seeds, len(local_model_updates), num_classes))
        return DDifs

    def _calculate_cosine_distances(self, local_model_updates: List[ModelUpdate], last_layer_name) -> np.ndarray:
        """Calculate cosine distances between client updates."""
        N = len(local_model_updates)
        bias_name = last_layer_name + ".bias"

        # Stack all bias vectors into a matrix (N x bias_dim)
        bias_vectors = torch.stack([
            local_model_updates[i][bias_name].flatten() 
            for i in range(N)
        ])

        # Normalize all vectors at once
        norms = torch.linalg.norm(bias_vectors, dim=1, keepdim=True)
        normalized_vectors = bias_vectors / (norms + 1e-10)

        # Compute all pairwise cosine similarities using matrix multiplication
        # similarity[i, j] = dot(normalized_vectors[i], normalized_vectors[j])
        similarities = torch.mm(normalized_vectors, normalized_vectors.t())

        # Convert similarities to distances with float64 for hdbscan compatibility
        distances = (1.0 - similarities).cpu().numpy().astype(np.float64)

        return distances

    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Calculate distance matrix from cluster assignments (following reference)."""
        pairwise_dists = np.ones((N, N))
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i][j] = 0
        return pairwise_dists

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
