"""
Multi-Metrics server implementation for FL.

This implements the Multi-Metrics aggregation algorithm proposed in
https://github.com/siquanhuang/Multi-metrics. The method combines multiple
distance metrics to score client updates and aggregates the most consistent
subset, providing robustness against adversarial updates.
"""

import numpy as np

from .anomaly_detection_server import AnomalyDetectionServer
from backfed.utils.logging_utils import log
from logging import INFO, WARNING

class MultiMetricsServer(AnomalyDetectionServer):
    """
    Server that implements the Multi-Metrics aggregation rule.

    The algorithm evaluates client updates using a mixture of cosine, L1 and
    norm-distance measurements. It then scores clients via a Mahalanobis
    distance in this metric space and aggregates the lowest-scoring (most
    consistent) fraction of updates.
    """

    def __init__(
        self,
        server_config,
        server_type: str = "multi_metrics",
        selection_ratio: float = 0.5, # p=0.3 is the default in the paper
        eta: float = 0.5,
    ):
        """
        Initialize the Multi-Metrics server.

        Args:
            server_config: Hydra server configuration.
            server_type: Name used to register the server.
            p: Fraction of clients to keep during aggregation.
                matrix inversion used in the Mahalanobis distance.
        """
        super(MultiMetricsServer, self).__init__(server_config, server_type, eta)
        self.selection_ratio = selection_ratio
        log(
            INFO,
            f"Initialized Multi-Metrics server with selection_ratio={self.selection_ratio}, "
        )

    def detect_anomalies(self, client_updates):
        """
        Detect anomalies in client updates using the Multi-Metrics algorithm.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_state_dict).
        Returns:
            True if aggregation succeeded, False otherwise.
        """
        if len(client_updates) == 0:
            return False

        # We need at least two clients to compute pairwise distances.
        if len(client_updates) == 1:
            log(WARNING, "Only one client update received, skipping Multi-Metrics aggregation")
            return super().aggregate_client_updates(client_updates)

        num_dps = []
        client_ids = []
        vectorize_nets = []

        global_params_np = self.global_parameters_vector.detach().cpu().numpy()
        for cid, num, update in client_updates:
            client_ids.append(cid)
            num_dps.append(num)
            client_update_np = self.parameters_dict_to_vector(update).detach().cpu().numpy()
            vectorize_nets.append(client_update_np + global_params_np)

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        log(INFO, f"Multi-Metrics scores: {[(cid, round(float(score), 2)) for cid, score in zip(client_ids, scores)]}")

        p_num = int(self.selection_ratio * len(scores))
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort
        selected_num_dps = np.array(num_dps)[topk_ind]

        benign_client_ids = [client_ids[ti] for ti in topk_ind]
        malicious_client_ids = [cid for cid in client_ids if cid not in benign_client_ids]

        log(INFO, "Num data points: {}".format(num_dps))
        log(INFO, "Num selected data points: {}".format(selected_num_dps))
                                                                                       
        return malicious_client_ids, benign_client_ids
