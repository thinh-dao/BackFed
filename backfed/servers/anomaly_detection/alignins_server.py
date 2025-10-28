"""
Implementation of AlignIns anomaly detection defense for federated learning.
Reference: https://github.com/JiiahaoXU/AlignIns
"""

import numpy as np
import torch
import torch.nn.functional as F

from .anomaly_detection_server import AnomalyDetectionServer
from typing import List, Tuple
from logging import INFO, WARNING
from backfed.utils import log
from backfed.const import ModelUpdate, client_id, num_examples

class AlignInsServer(AnomalyDetectionServer):
    """
    AlignIns server that filters malicious updates using TDA/MPSA statistics and
    applies norm-clipped aggregation on the remaining benign updates.
    """

    def __init__(
        self,
        server_config,
        sparsity: float = 0.3,
        lambda_s: float = 1.0,
        lambda_c: float = 1.0,
        eta: float = 0.5,
        server_type: str = "alignins",
        **kwargs,
    ) -> None:
        super().__init__(server_config, server_type=server_type, eta=eta, **kwargs)
        self.sparsity = float(sparsity)
        self.lambda_s = float(lambda_s)
        self.lambda_c = float(lambda_c)

        log(
            INFO,
            (
                f"Initialized AlignIns server with sparsity={self.sparsity}, "
                f"lambda_s={self.lambda_s}, lambda_c={self.lambda_c}, eta={self.eta}"
            ),
        )

    def detect_anomalies(
        self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect anomalous updates based on MPSA and TDA MZ-scores.
        """
        local_updates = []
        chosen_clients = []

        for client_id, _, client_update in client_updates:
            local_updates.append(self.parameters_dict_to_vector(client_update))
            chosen_clients.append(client_id)

        num_chosen_clients = len(chosen_clients)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            # compute top-k indices based on configured sparsity (use instance attribute)
            abs_update = torch.abs(inter_model_updates[i])
            k = max(int(abs_update.numel() * self.sparsity), 1)
            k = min(k, abs_update.numel())
            if k >= abs_update.numel():
                init_indices = torch.arange(abs_update.numel(), device=abs_update.device, dtype=torch.long)
            else:
                init_indices = torch.topk(abs_update, k, largest=True).indices

            mpsa_list.append(
                (
                    torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices])
                    / torch.numel(inter_model_updates[i][init_indices])
                ).item()
            )

            tda_list.append(cos(inter_model_updates[i], self.global_parameters_vector).item())

        log(INFO, f'AlignIns TDA: {[round(i, 4) for i in tda_list]}')
        log(INFO, f'AlignIns MPSA: {[round(i, 4) for i in mpsa_list]}')

        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        log(INFO, f'AlignIns MZ-score of MPSA: {[(cid, round(float(score), 4)) for cid, score in zip(chosen_clients, mzscore_mpsa)]}')
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        log(INFO, f'AlignIns MZ-score of TDA: {[(cid, round(float(score), 4)) for cid, score in zip(chosen_clients, mzscore_tda)]}')

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set(range(num_chosen_clients)).intersection(
            {int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.lambda_s)}
        )
        benign_idx2 = set(range(num_chosen_clients)).intersection(
            {int(i) for i in np.argwhere(np.array(mzscore_tda) < self.lambda_c)}
        )

        benign_set = benign_idx1.intersection(benign_idx2)
        benign_idx = sorted(list(benign_set))

        benign_clients = [chosen_clients[i] for i in benign_idx]
        malicious_clients = [chosen_clients[i] for i in range(num_chosen_clients) if i not in benign_set]

        # compute euclidean distances (L2) of benign updates to the global vector
        euclidean_distances = [
            torch.linalg.norm(local_updates[i] - self.global_parameters_vector, ord=2).item()
            for i in benign_idx
        ]

        return malicious_clients, benign_clients, euclidean_distances

    def aggregate_client_updates(
        self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]
    ):
        """
        Apply AlignIns detection followed by norm-clipped weighted aggregation.
        """
        if len(client_updates) == 0:
            log(WARNING, "AlignIns: No client updates found.")
            return False

        # Evaluate detection and log metrics
        malicious_clients, benign_clients, euclidean_distances = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        detection_metrics = self.evaluate_detection(benign_clients, malicious_clients, true_malicious_clients, len(client_updates))

        # If no benign clients were found, skip update
        if len(benign_clients) == 0:
            log(WARNING, "AlignIns: no benign clients identified, skipping model update.")
            return False

        # Aggregate clipped differences from benign clients
        clip_norm = torch.median(torch.tensor(euclidean_distances)).item()

        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model.state_dict().items()
        }

        # map benign client id -> distance for correct lookup
        benign_dist_map = {cid: d for cid, d in zip(benign_clients, euclidean_distances)}

        weight = 1.0 / len(benign_clients)
        global_state_dict = self.global_model.state_dict()
        for client_id, num_examples, update in client_updates:
            if client_id in malicious_clients:
                continue

            dist = benign_dist_map[client_id]

            for name, param in update.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                diff = (param.to(self.device) - global_state_dict[name])
                if dist > clip_norm and clip_norm > 0:
                    diff *= clip_norm / dist
                weight_accumulator[name].add_(diff * weight)

        # Update global model
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

        return True
