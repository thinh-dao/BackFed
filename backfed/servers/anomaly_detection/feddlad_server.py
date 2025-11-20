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
        bg: int = 4,
        pg: int = 1,
        iqr_scale: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(server_config, server_type, eta, **kwargs)
        self.bg = bg
        self.pg = pg
        self.iqr_scale = iqr_scale
        log(INFO, f"Initialized FedDLAD server with bg={self.bg}, pg={self.pg}, iqr_scale={self.iqr_scale}")

    @torch.no_grad()
    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, ModelUpdate]],
    ) -> bool:
        if not client_updates:
            log(WARNING, "FedDLAD: No client updates received; skipping aggregation.")
            return False

        # Prepare client data
        global_vector = self.global_parameters_vector.detach().to(self.device)
        agent_updates = {cid: self.parameters_dict_to_vector(update) for cid, _, update in client_updates}
        agent_parameters = {cid: global_vector + agent_updates[cid] for cid in agent_updates}
        agent_full_updates = {cid: update for cid, _, update in client_updates}

        # Detect anomalies and aggregate
        aggregated_update, benign_clients, malicious_clients = self._detect_and_aggregate(
            agent_updates, agent_parameters
        )

        # Evaluate detection performance
        true_malicious = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(benign_clients, malicious_clients, true_malicious, len(client_updates))

        if aggregated_update is None:
            log(WARNING, "FedDLAD: Aggregated update is None; skipping model update.")
            return False

        # Apply updates
        self._apply_updates(aggregated_update, agent_full_updates, benign_clients)
        
        return True

    def _apply_updates(
        self,
        update_vector: torch.Tensor,
        agent_full_updates: Dict[int, ModelUpdate],
        benign_clients: List[int],
    ) -> None:
        """Apply aggregated vector update and batch normalization parameters to global model."""
        # Apply trainable parameters from vector update
        offset = 0
        for name, param in self.global_model.named_parameters():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            numel = param.numel()
            delta = update_vector[offset : offset + numel].view_as(param)
            param.data.add_(delta.to(param.device) * self.eta)
            offset += numel

        # Apply batch normalization parameters (running_mean, running_var)
        if not benign_clients:
            return
        
        num_benign = len(benign_clients)
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue

            if "running_mean" in name or "running_var" in name:
                # Aggregate from benign clients only
                bn_update = sum(agent_full_updates[cid][name].to(self.device, dtype=torch.float32) 
                                for cid in benign_clients) / num_benign
                param.data.add_(bn_update * self.eta)

    def _detect_and_aggregate(
        self,
        agent_updates: Dict[int, torch.Tensor],
        agent_parameters: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor | None, List[int], List[int]]:
        """Main FedDLAD detection and aggregation pipeline."""
        if not agent_updates:
            return None, [], []

        # Clone updates for processing
        updates = {cid: update.clone() for cid, update in agent_updates.items()}
        
        # Step 1: Select reference clients using COF
        reference_ids = self._select_reference_clients(agent_parameters)

        # Step 2: Apply norm scaling and IQR outlier flipping
        self._apply_norm_scaling(updates, reference_ids)
        self._flip_iqr_outliers(updates)

        # Step 3: Compute reference update (unweighted average)
        reference_update = torch.mean(torch.stack([updates[cid] for cid in reference_ids]), dim=0)
        
        # Step 4: Secondary filtering to pardon additional clients
        pardoned_ids, score_dict = self._secondary_filter(reference_ids, reference_update, updates)
        if len(pardoned_ids) == 0:
            return reference_update, reference_ids, []
        
        log(INFO, f"FedDLAD: pardoned clients={pardoned_ids} with scores={ {cid: score_dict[cid] for cid in pardoned_ids} }")
        
        # Step 5: Mix reference and pardoned updates
        pardoned_update = self._weighted_pardoned_average(pardoned_ids, updates, score_dict)
        n_ref, n_pard = len(reference_ids), len(pardoned_ids)
        final_update = (n_ref * reference_update + n_pard * pardoned_update) / (n_ref + n_pard)    

        # Classify clients
        benign_clients = sorted(set(reference_ids + pardoned_ids))
        malicious_clients = [cid for cid in updates.keys() if cid not in benign_clients]

        log(INFO, f"FedDLAD: reference={reference_ids}, pardoned={pardoned_ids}")
        return final_update, benign_clients, malicious_clients

    def _select_reference_clients(self, agent_parameters: Dict[int, torch.Tensor]) -> List[int]:
        """Select top-bg clients with lowest COF anomaly scores."""
        client_ids = list(agent_parameters.keys())
        if len(client_ids) <= 1:
            return client_ids

        # Convert to numpy for COF
        parameter_matrix = torch.stack([agent_parameters[cid] for cid in client_ids]).detach().cpu().numpy().astype(np.float64)
        
        # Compute COF scores
        cosine_distance = 1.0 - cosine_similarity(parameter_matrix)
        cof = COF(contamination=max(1.0 / len(client_ids), 1e-3), n_neighbors=max(len(client_ids) - 1, 1))
        cof.fit(cosine_distance)
        scores = cof.decision_function(cosine_distance)

        # Return top-bg clients with lowest scores
        ranked = sorted(zip(client_ids, scores), key=lambda x: x[1])
        return [cid for cid, _ in ranked[:min(self.bg, len(ranked))]]

    def _apply_norm_scaling(self, updates: Dict[int, torch.Tensor], reference_ids: List[int]) -> None:
        """Clip update norms to the median norm of reference clients."""
        if not reference_ids:
            return

        # Compute median norm from reference clients
        norms = [torch.linalg.norm(updates[cid].detach().cpu()).item() for cid in reference_ids]
        median_norm = np.median(norms)
        
        if median_norm == 0:
            return

        # Scale updates exceeding median norm
        for cid, update in updates.items():
            update_norm = torch.linalg.norm(update.detach().cpu()).item()
            if update_norm > median_norm:
                updates[cid] = update * (median_norm / update_norm)

    def _flip_iqr_outliers(self, updates: Dict[int, torch.Tensor]) -> None:
        """Flip per-dimension outliers based on IQR."""
        if len(updates) <= 1 or self.iqr_scale == 0:
            return

        # Compute IQR bounds using numpy percentile
        all_updates = np.array([update.clone().cpu().numpy() for update in updates.values()])
        q1 = np.percentile(all_updates, 25, axis=0)
        q3 = np.percentile(all_updates, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_scale * iqr
        upper_bound = q3 + self.iqr_scale * iqr

        # Flip outliers dimension-wise
        for cid, update in updates.items():
            update_vector = update.clone().cpu().numpy()
            mask = (update_vector < lower_bound) | (update_vector > upper_bound)
            if mask.any():
                update_vector[mask] = -update_vector[mask]
                updates[cid] = torch.from_numpy(update_vector).float().to(self.device)

    def _secondary_filter(
        self,
        reference_ids: List[int],
        reference_update: torch.Tensor,
        updates: Dict[int, torch.Tensor],
    ) -> Tuple[List[int], Dict[int, float]]:
        """Select top-pg clients most similar to reference update via cosine similarity."""
        # Compute cosine similarity scores for non-reference clients
        ref_np = reference_update.cpu().numpy().reshape(1, -1)
        score_dict = {}
        for cid, update in updates.items():
            if cid not in reference_ids:
                update_np = update.cpu().numpy().reshape(1, -1)
                sim_cosine = cosine_similarity(ref_np, update_np)[0][0]
                score_dict[cid] = max(sim_cosine, 0.0)  # ReLU
        
        # Filter out zero scores and select top-pg
        filtered_scores = {cid: score for cid, score in score_dict.items() if score > 0}
        if not filtered_scores:
            return [], score_dict
        
        pardoned_ids = sorted(filtered_scores, key=filtered_scores.get, reverse=True)[:self.pg]
        return pardoned_ids, score_dict

    def _weighted_pardoned_average(
        self,
        pardoned_ids: List[int],
        updates: Dict[int, torch.Tensor],
        score_dict: Dict[int, float],
    ) -> torch.Tensor:
        """Compute weighted average of pardoned updates by their similarity scores."""
        pardoned_update = sum(score_dict[cid] * updates[cid] for cid in pardoned_ids)
        total_score = sum(score_dict[cid] for cid in pardoned_ids)
        return pardoned_update / total_score if total_score > 0 else torch.mean(torch.stack([updates[cid] for cid in pardoned_ids]), dim=0)
