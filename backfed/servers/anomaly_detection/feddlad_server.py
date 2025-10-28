"""
FedDLAD anomaly detection and aggregation defense.

Reference implementation: https://github.com/dingbinb/FedDLAD
"""

from .anomaly_detection_server import AnomalyDetectionServer
from logging import INFO, WARNING
from pyod.models.cof import COF
from sklearn.metrics.pairwise import cosine_similarity
from backfed.const import ModelUpdate, client_id, num_examples
from backfed.utils import log
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F



class FedDLADServer(AnomalyDetectionServer):
    """
    Implementation of the FedDLAD server-side anomaly detection defense.

    FedDLAD performs:
      1. COF-based reference client selection.
      2. Norm clipping relative to the reference group.
      3. IQR-based per-dimension outlier flipping.
      4. Secondary filtering that pardons additional clients aligned with the reference update.
    """

    def __init__(
        self,
        server_config,
        server_type: str = "feddlad",
        eta: float = 0.5,
        bg: int = 12,
        pg: int = 3,
        iqr_scale: float = 0.6,
        cof_contamination: float = 0.1,
        cof_neighbors: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(server_config, server_type, eta, **kwargs)
        self.bg = max(1, bg)
        self.pg = max(0, pg)
        self.iqr_scale = max(0.0, iqr_scale)
        self.cof_contamination = max(1e-3, min(cof_contamination, 0.5 - 1e-3))
        self.cof_neighbors = max(1, cof_neighbors)
        log(INFO, f"Initialized FedDLAD server with bg={self.bg}, pg={self.pg}, iqr_scale={self.iqr_scale}")

    @torch.no_grad()
    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, ModelUpdate]],
    ) -> bool:
        if not client_updates:
            log(WARNING, "FedDLAD: No client updates received; skipping aggregation.")
            return False

        global_vector = self.global_parameters_vector.detach().to(self.device)
        agent_updates: Dict[int, torch.Tensor] = {}
        agent_parameters: Dict[int, torch.Tensor] = {}
        client_sizes: Dict[int, int] = {}

        for cid, num_samples, state_dict in client_updates:
            client_vector = self._state_dict_to_vector(state_dict)
            update_vector = client_vector - global_vector
            agent_updates[cid] = update_vector
            agent_parameters[cid] = client_vector
            client_sizes[cid] = max(1, int(num_samples))

        aggregated_update, benign_clients, malicious_clients = self._combined_aggregation(
            agent_updates, agent_parameters, client_sizes
        )

        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(
            benign_clients=benign_clients,
            malicious_clients=malicious_clients,
            true_malicious_clients=true_malicious_clients,
            total_updates=len(client_updates),
        )

        if aggregated_update is None:
            log(WARNING, "FedDLAD: Aggregated update is None; skipping model update.")
            return False

        self._apply_vector_update(aggregated_update)
        return True

    def _state_dict_to_vector(self, state_dict: ModelUpdate) -> torch.Tensor:
        """Flatten client state dict into a vector aligned with global parameters."""
        vectors = []
        for name, param in self.global_model.named_parameters():
            client_param = state_dict[name].to(self.device, dtype=param.dtype)
            vectors.append(client_param.view(-1))
        return torch.cat(vectors).detach()

    def _apply_vector_update(self, update_vector: torch.Tensor) -> None:
        """Apply flattened update vector to the global model parameters."""
        offset = 0
        for name, param in self.global_model.named_parameters():
            numel = param.numel()
            delta = update_vector[offset : offset + numel].view_as(param)
            param.data.add_(delta.to(param.device) * self.eta)
            offset += numel

    def _combined_aggregation(
        self,
        agent_updates: Dict[int, torch.Tensor],
        agent_parameters: Dict[int, torch.Tensor],
        client_sizes: Dict[int, int],
    ) -> Tuple[torch.Tensor | None, List[int], List[int]]:
        if not agent_updates:
            return None, [], []

        updates = {cid: update.clone() for cid, update in agent_updates.items()}
        reference_ids = self._select_reference_clients(agent_parameters)

        if not reference_ids:
            reference_ids = list(updates.keys())
            log(WARNING, "FedDLAD: COF returned no reference clients; defaulting to all clients.")

        self._apply_norm_scaling(updates, reference_ids)
        self._flip_iqr_outliers(updates)

        reference_update = self._weighted_average(reference_ids, updates, client_sizes)
        if reference_update is None:
            reference_update = torch.mean(torch.stack(list(updates.values())), dim=0)

        pardoned_ids, score_dict = self._secondary_filter(reference_ids, reference_update, updates)
        total_update = self._mix_updates(reference_ids, pardoned_ids, reference_update, updates, score_dict)

        benign_clients = sorted(set(reference_ids + pardoned_ids))
        malicious_clients = [cid for cid in updates.keys() if cid not in benign_clients]

        log(INFO, f"FedDLAD: reference={reference_ids}, pardoned={pardoned_ids}, benign={benign_clients}")
        return total_update, benign_clients, malicious_clients

    def _select_reference_clients(self, agent_parameters: Dict[int, torch.Tensor]) -> List[int]:
        client_ids = list(agent_parameters.keys())
        if len(client_ids) <= 1:
            return client_ids

        matrix = torch.stack(
            [agent_parameters[cid].detach().cpu().to(torch.float64) for cid in client_ids],
            dim=0,
        ).numpy()

        cosine_distance = 1.0 - cosine_similarity(matrix)
        n_neighbors = min(self.cof_neighbors, max(len(client_ids) - 1, 1))
        contamination = min(self.cof_contamination, max(1.0 / len(client_ids), 1e-3))

        cof = COF(contamination=contamination, n_neighbors=n_neighbors)
        cof.fit(cosine_distance)
        scores = cof.decision_function(cosine_distance)

        ranked = sorted(zip(client_ids, scores), key=lambda item: item[1])
        top_k = min(self.bg, len(ranked))
        return [cid for cid, _ in ranked[:top_k]]

    def _apply_norm_scaling(self, updates: Dict[int, torch.Tensor], reference_ids: List[int]) -> None:
        if not reference_ids:
            return

        reference_norms = torch.tensor(
            [updates[cid].norm().item() for cid in reference_ids],
            device=self.device,
        )
        median_norm = torch.median(reference_norms)

        if median_norm.item() == 0:
            return

        for cid, update in updates.items():
            norm = update.norm()
            if norm.item() > median_norm.item() and norm.item() > 0:
                scale = median_norm / norm
                updates[cid] = update * scale

    def _flip_iqr_outliers(self, updates: Dict[int, torch.Tensor]) -> None:
        if len(updates) <= 1 or self.iqr_scale == 0:
            return

        stacked = torch.stack([update.detach().cpu() for update in updates.values()], dim=0)
        q1 = torch.quantile(stacked, 0.25, dim=0)
        q3 = torch.quantile(stacked, 0.75, dim=0)
        iqr = q3 - q1

        lower = q1 - self.iqr_scale * iqr
        upper = q3 + self.iqr_scale * iqr
        lower = lower.to(self.device)
        upper = upper.to(self.device)

        for cid, update in updates.items():
            mask = (update < lower) | (update > upper)
            if mask.any():
                updates[cid] = torch.where(mask, -update, update)

    def _weighted_average(
        self,
        ids: List[int],
        updates: Dict[int, torch.Tensor],
        client_sizes: Dict[int, int],
    ) -> torch.Tensor | None:
        if not ids:
            return None

        device = next(iter(updates.values())).device
        total_weight = 0.0
        accumulator = torch.zeros_like(next(iter(updates.values())), device=device)
        for cid in ids:
            weight = float(client_sizes.get(cid, 0))
            total_weight += weight
            accumulator += updates[cid] * weight

        if total_weight > 0:
            return accumulator / total_weight
        return torch.mean(torch.stack([updates[cid] for cid in ids]), dim=0)

    def _secondary_filter(
        self,
        reference_ids: List[int],
        reference_update: torch.Tensor,
        updates: Dict[int, torch.Tensor],
    ) -> Tuple[List[int], Dict[int, float]]:
        if not reference_ids:
            return [], {}

        ref_norm = reference_update.norm().item()
        if ref_norm == 0:
            return [], {}

        scores: Dict[int, float] = {}
        reference_flat = reference_update.view(1, -1)
        for cid, update in updates.items():
            if cid in reference_ids:
                continue
            update_flat = update.view(1, -1)
            similarity = F.cosine_similarity(reference_flat, update_flat, dim=1, eps=1e-12).item()
            score = max(similarity, 0.0)
            if score > 0:
                scores[cid] = score

        if not scores:
            return [], {}

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_k = min(self.pg, len(ranked))
        pardoned_ids = [cid for cid, _ in ranked[:top_k]]
        return pardoned_ids, scores

    def _mix_updates(
        self,
        reference_ids: List[int],
        pardoned_ids: List[int],
        reference_update: torch.Tensor,
        updates: Dict[int, torch.Tensor],
        score_dict: Dict[int, float],
    ) -> torch.Tensor:
        if not reference_ids:
            if not pardoned_ids:
                stacked = torch.stack(list(updates.values()))
                return torch.mean(stacked, dim=0)
            return self._average_pardoned_updates(pardoned_ids, updates, score_dict)

        total_count = len(reference_ids) + len(pardoned_ids)
        if total_count == 0:
            stacked = torch.stack(list(updates.values()))
            return torch.mean(stacked, dim=0)

        total_update = reference_update * (len(reference_ids) / total_count)

        if pardoned_ids:
            pardoned_update = self._average_pardoned_updates(pardoned_ids, updates, score_dict)
            total_update = total_update + pardoned_update * (len(pardoned_ids) / total_count)

        return total_update

    def _average_pardoned_updates(
        self,
        pardoned_ids: List[int],
        updates: Dict[int, torch.Tensor],
        score_dict: Dict[int, float],
    ) -> torch.Tensor:
        device = next(iter(updates.values())).device
        accumulator = torch.zeros_like(next(iter(updates.values())), device=device)
        total_score = 0.0
        for cid in pardoned_ids:
            score = float(score_dict.get(cid, 0.0))
            accumulator += updates[cid] * score
            total_score += score

        if total_score > 0:
            return accumulator / total_score
        return torch.mean(torch.stack([updates[cid] for cid in pardoned_ids]), dim=0)
