"""
Defense category base classes for federated learning.
"""
import wandb
import warnings
import torch
import numpy as np
# Suppress specific scikit-learn deprecation warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

from sklearn.cluster import KMeans
from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.utils import log
from fl_bdbench.const import client_id, num_examples, StateDict
from typing import List, Tuple, Dict
from logging import INFO

class ClientSideDefenseServer(UnweightedFedAvgServer):
    """Base class for all client-side defenses.

    Client-side defenses operate during client training by modifying the client's
    training process, objective function, or update mechanism before sending to the server.
    """
    defense_categories = ["client_side"]

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)


class RobustAggregationServer(UnweightedFedAvgServer):
    """Base class for all robust aggregation defenses.

    Robust aggregation defenses modify the aggregation algorithm to be resilient
    against malicious updates, typically by using robust statistics.
    """
    defense_categories = ["robust_aggregation"]

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)


MaliciousClientsIds = List[int]
BenignClientsIds = List[int]

class AnomalyDetectionServer(UnweightedFedAvgServer):
    """Base class for all anomaly detection defenses.

    Anomaly detection defenses identify and filter malicious updates by detecting
    statistical anomalies or patterns indicative of attacks.
    """
    defense_categories = ["anomaly_detection"]

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)
        # Initialize detection performance metrics
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]], **kwargs) -> Tuple[MaliciousClientsIds, BenignClientsIds]:
        """
        Detect anomalies in the updates. This method should be overridden by defenses.

        Args:
            client_updates: List of client updates to check for anomalies
            **kwargs: Additional arguments for detection 

        Returns:
            Tuple of lists:
            - List of client_ids classified as anomalous
            - List of client_ids classified as benign
        """
        pass
    
    def gap_statistics(self, data: np.ndarray, num_sampling: int, K_max: int, n: int) -> int:
        """Implement gap statistics for optimal cluster number selection.
        
        Note: We convert to numpy for sklearn compatibility, but use PyTorch for preprocessing.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy() # Convert to numpy for sklearn compatibility
        
        # Reshape data
        data = data.reshape(data.shape[0], -1)
        
        # Normalize data (min-max scaling)
        data_c = np.zeros_like(data)
        for i in range(data.shape[1]):
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            if max_val > min_val:
                data_c[:, i] = (data[:, i] - min_val) / (max_val - min_val)
            else:
                data_c[:, i] = 0.0

        gaps = []
        s_values = []

        for k in range(1, K_max + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=self.config.seed).fit(data_c)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Calculate within-cluster dispersion
            wk = 0
            for i in range(k):
                cluster_points = data_c[labels == i]
                if len(cluster_points) > 0:
                    wk += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1))

            # Calculate expected dispersion for random data
            wkbs = []
            for _ in range(num_sampling):
                random_data = np.random.uniform(0, 1, size=(n, data_c.shape[1]))
                kmeans_b = KMeans(n_clusters=k, init='k-means++', random_state=self.config.seed).fit(random_data)
                wkb = 0
                for i in range(k):
                    cluster_points = random_data[kmeans_b.labels_ == i]
                    if len(cluster_points) > 0:
                        wkb += np.sum(np.linalg.norm(cluster_points - kmeans_b.cluster_centers_[i], axis=1))
                wkbs.append(np.log(wkb + 1e-10))  # Add small epsilon to avoid log(0)

            # Calculate gap statistic
            gap = np.mean(wkbs) - np.log(wk + 1e-10)  # Add small epsilon to avoid log(0)
            sd = np.std(wkbs) * np.sqrt(1 + 1/num_sampling)

            gaps.append(gap)
            s_values.append(sd)

        # Find optimal number of clusters
        for k in range(1, K_max):
            if gaps[k-1] >= gaps[k] - s_values[k]:
                return k
        return K_max
    
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
        malicious_clients, benign_clients = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        detection_metrics = self.evaluate_detection(malicious_clients, true_malicious_clients, len(client_updates))
        
        # Aggregate benign updates
        benign_updates = [client_update for client_update in client_updates if client_update[0] in benign_clients]
        return super().aggregate_client_updates(benign_updates)

    def evaluate_detection(self, malicious_clients: List[int], true_malicious_clients: List[int], total_updates: int):
        """
        Evaluate detection performance by comparing detected anomalies with ground truth.

        Args:
            malicious_clients: List of indices that were detected as anomalous
            true_malicious_clients: List of indices that are actually malicious (ground truth)
            total_updates: Total number of updates being evaluated 

        Returns:
            Dictionary with precision, recall, F1, and FPR for this round.
        """
        detected_set = set(malicious_clients)
        true_set = set(true_malicious_clients)

        log(INFO, f"═══ {self.__class__.__name__} detection results ═══")
        log(INFO, f"Predicted malicious clients: {list(detected_set)}")
        log(INFO, f"Ground-truth malicious clients: {list(true_set)}")

        # Calculate metrics for this round
        tp = len(detected_set.intersection(true_set))
        fp = len(detected_set - true_set)
        fn = len(true_set - detected_set)
        tn = total_updates - tp - fp - fn

        # Update cumulative metrics based on whether malicious clients are present
        if len(true_malicious_clients) == 0:
            # Track metrics for clean rounds separately
            if not hasattr(self, 'clean_rounds'):
                self.clean_rounds = 0
                self.clean_false_positives = 0
                self.clean_true_negatives = 0
            
            self.clean_rounds += 1
            self.clean_false_positives += fp
            self.clean_true_negatives += tn
        else:
            # Update standard metrics for rounds with malicious clients
            self.true_positives += tp
            self.false_positives += fp
            self.true_negatives += tn
            self.false_negatives += fn

        # Calculate key metrics for this round
        precision = tp / max(tp + fp, 1)  # Precision = TP / (TP + FP)
        recall = tp / max(tp + fn, 1)     # Recall = TP / (TP + FN)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)  # F1 score
        fpr = fp / max(fp + tn, 1)        # False Positive Rate = FP / (FP + TN)

        if len(true_malicious_clients) == 0:
            detection_metrics = {
                "fpr_clean": fpr,  # If no malicious clients, we only want to measure false alarm rate
            }
        else:
            detection_metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
            }
                
        log(INFO, detection_metrics)
        log(INFO, f"═══════════════════════════════════════════════")

        if self.config.save_logging in ["wandb", "both"]:
            wandb.log({**detection_metrics}, step=self.current_round)
        elif self.config.save_logging in ["csv", "both"]:
            self.csv_logger.log({**detection_metrics}, step=self.current_round)
        return detection_metrics

    def get_detection_performance(self):
        """
        Get overall detection performance metrics.

        Returns:
            Dictionary with the most important detection metrics:
            - precision: Percentage of detected anomalies that are actually malicious
            - recall (TPR): Percentage of actual malicious updates that were detected
            - f1_score: Harmonic mean of precision and recall
            - fpr: Percentage of benign updates incorrectly flagged as malicious
            - fpr_clean: False positive rate for rounds with no malicious clients
        """
        # Calculate metrics for rounds with malicious clients
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr = self.false_positives / max(self.false_positives + self.true_negatives, 1)
        
        # Calculate FPR for clean rounds
        fpr_clean = 0.0
        if hasattr(self, 'clean_rounds') and self.clean_rounds > 0:
            fpr_clean = self.clean_false_positives / max(self.clean_false_positives + self.clean_true_negatives, 1)

        # Return focused set of metrics
        result = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
        }
        
        # Add clean FPR if we have clean rounds
        if hasattr(self, 'clean_rounds') and self.clean_rounds > 0:
            result["fpr_clean"] = fpr_clean
        
        return result

    def reset_detection_metrics(self):
        """Reset all detection performance metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # Reset clean round metrics if they exist
        if hasattr(self, 'clean_rounds'):
            self.clean_rounds = 0
            self.clean_false_positives = 0
            self.clean_true_negatives = 0

class PostAggregationServer(UnweightedFedAvgServer):
    """Base class for all post-aggregation defenses.

    Post-aggregation defenses apply additional processing after the initial aggregation
    to further mitigate the impact of malicious updates.
    """
    defense_categories = ["post_aggregation"]

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)

# For Hybrid defenses, we can do multiple inheritance (see FlameServer)
