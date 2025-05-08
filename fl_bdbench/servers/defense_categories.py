"""
Defense category base classes for federated learning.
"""
import wandb

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from typing import List, Tuple, Dict
from logging import WARNING, log

class ClientSideDefenseServer(UnweightedFedAvgServer):
    """Base class for all client-side defenses.

    Client-side defenses operate during client training by modifying the client's
    training process, objective function, or update mechanism before sending to the server.
    """
    defense_category = "client_side"

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)


class RobustAggregationServer(BaseServer):
    """Base class for all robust aggregation defenses.

    Robust aggregation defenses modify the aggregation algorithm to be resilient
    against malicious updates, typically by using robust statistics.
    """
    defense_category = "robust_aggregation"

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)

class AnomalyDetectionServer(UnweightedFedAvgServer):
    """Base class for all anomaly detection defenses.

    Anomaly detection defenses identify and filter malicious updates by detecting
    statistical anomalies or patterns indicative of attacks.
    """
    defense_category = "anomaly_detection"

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)
        # Initialize detection performance metrics
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def detect_anomalies(self, updates, **kwargs):
        """
        Detect anomalies in the updates.

        Args:
            updates: List of client updates to check for anomalies
            **kwargs: Additional arguments for detection 

        Returns:
            List of indices of updates classified as anomalous
        """
        pass

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]):
        """
        Standard AnomalyDetectionServer procedure: Find malicious clients, evaluate detection, and aggregate benign updates.
        """
        if len(client_updates) == 0:
            return False
        
        # Detect anomalies & evaluate detection
        predicted_malicious_client_indices = self.detect_anomalies(client_updates)
        true_malicious_client_indices = self.get_clients_info(self.current_round)["malicious_clients"]
        detection_metrics = self.evaluate_detection(predicted_malicious_client_indices, true_malicious_client_indices, len(client_updates))

        # Log detection metrics
        if self.config.save_logging in ["wandb", "both"]:
            wandb.log({**detection_metrics}, step=self.current_round)
        elif self.config.save_logging in ["csv", "both"]:
            self.csv_logger.log({**detection_metrics}, step=self.current_round)
        
        # Aggregate benign updates
        benign_client_indices = [i for i in range(len(client_updates)) if i not in predicted_malicious_client_indices]
        benign_updates = [client_updates[i] for i in benign_client_indices]
        return super().aggregate_client_updates(benign_updates)

    def evaluate_detection(self, predicted_malicious_indices: List[int], true_malicious_indices: List[int], total_updates: int):
        """
        Evaluate detection performance by comparing detected anomalies with ground truth.

        Args:
            predicted_malicious_indices: List of indices that were detected as anomalous
            true_malicious_indices: List of indices that are actually malicious (ground truth)
            total_updates: Total number of updates being evaluated 

        Returns:
            Dictionary with precision, recall, F1, and FPR for this round.
        """
        detected_set = set(predicted_malicious_indices)
        true_set = set(true_malicious_indices)

        # Calculate metrics for this round
        tp = len(detected_set.intersection(true_set))
        fp = len(detected_set - true_set)
        fn = len(true_set - detected_set)
        tn = total_updates - tp - fp - fn

        # Update cumulative metrics
        self.true_positives += tp
        self.false_positives += fp
        self.true_negatives += tn
        self.false_negatives += fn

        # Calculate key metrics for this round
        precision = tp / max(tp + fp, 1)  # Precision = TP / (TP + FP)
        recall = tp / max(tp + fn, 1)     # Recall = TP / (TP + FN)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)  # F1 score
        fpr = fp / max(fp + tn, 1)        # False Positive Rate = FP / (FP + TN)

        if len(true_malicious_indices) == 0:
            return {
                "fpr": fpr,
            }
        else:
            # Combine detection metrics with any additional detection info
            result = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "fpr": fpr,
            }

        return result

    def get_detection_performance(self):
        """
        Get overall detection performance metrics.

        Returns:
            Dictionary with the most important detection metrics:
            - precision: Percentage of detected anomalies that are actually malicious
            - recall (TPR): Percentage of actual malicious updates that were detected
            - f1_score: Harmonic mean of precision and recall
            - fpr: Percentage of benign updates incorrectly flagged as malicious
        """
        if self.detection_rounds == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "fpr": 0.0,
                "rounds": 0
            }

        # Calculate key metrics
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        fpr = self.false_positives / max(self.false_positives + self.true_negatives, 1)

        # Return focused set of metrics
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
        }

    def reset_detection_metrics(self):
        """Reset all detection performance metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

class PostAggregationServer(UnweightedFedAvgServer):
    """Base class for all post-aggregation defenses.

    Post-aggregation defenses apply additional processing after the initial aggregation
    to further mitigate the impact of malicious updates.
    """
    defense_category = "post_aggregation"

    def __init__(self, server_config, server_type, **kwargs):
        super().__init__(server_config, server_type, **kwargs)
