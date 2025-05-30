"""
FoolsGold server implementation.
Paper: https://www.usenix.org/conference/raid2020/presentation/fung
"""
import torch
from typing import Dict, List, Tuple
from logging import INFO

from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log
from backfed.const import StateDict, client_id, num_examples

class FoolsGoldServer(RobustAggregationServer):
    """
    FoolsGold server that uses cosine similarity to detect and defend against sybil attacks.
    """
    def __init__(self, server_config, server_type="foolsgold", confidence=1):
        super(FoolsGoldServer, self).__init__(server_config, server_type)
        self.confidence = confidence
        self.update_history: Dict[int, torch.Tensor] = {}  # client_id -> update_vector
        log(INFO, f"Initialized FoolsGold server with confidence={confidence}")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
        """
        Aggregate client updates using FoolsGold algorithm.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client IDs and their updates
        client_ids = [client_id for client_id, _, _ in client_updates]

        # Update history for each client
        for client_id, _, client_params in client_updates:
            # Convert model updates to flat vector
            update_vector = []
            for name, param in client_params.items():
                if "bias" in name or "weight" in name:
                    diff = param.to(self.device) - self.global_model_params[name]
                    update_vector.append(diff.flatten())

            update_vector = torch.cat(update_vector)

            # Normalize update vector
            norm = torch.linalg.norm(update_vector)
            if norm > 1:
                update_vector = update_vector / norm

            # Update history
            if client_id not in self.update_history:
                self.update_history[client_id] = update_vector.cpu()
            else:
                self.update_history[client_id] += update_vector.cpu()

        # Calculate FoolsGold weights
        foolsgold_weights = self._foolsgold(client_ids)
        log(INFO, f"FoolsGold weights (client_id, weight): {list(zip(client_ids, foolsgold_weights.tolist()))}")

        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model_params.items()
        }

        for weight, (cid, num_samples, client_state) in zip(foolsgold_weights, client_updates):
            for name, param in client_state.items():
                if name.endswith('num_batches_tracked'):
                    continue
                diff = param.to(self.device) - self.global_model_params[name]
                weight_accumulator[name].add_(diff * weight)

        # Update global model with learning rate
        for name, param in self.global_model_params.items():
            if "bias" in name or "weight" in name:
                param.add_(weight_accumulator[name])

        return True

    def _foolsgold(self, selected_clients) -> torch.Tensor:
        """
        Compute FoolsGold weights for the selected clients.

        Args:
            selected_clients: List of client IDs
        Returns:
            torch.Tensor of weights for each client
        """
        num_clients = len(selected_clients)
        selected_his = []

        for client_id in selected_clients:
            selected_his.append(self.update_history[client_id].to(self.device))

        # Stack client update histories
        M = torch.stack(selected_his)

        # Compute cosine similarity matrix
        cs_matrix = torch.zeros((num_clients, num_clients), device=self.device)
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:  # Skip diagonal (self-similarity)
                    # Compute single cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        selected_his[i].unsqueeze(0), 
                        selected_his[j].unsqueeze(0), 
                        dim=1
                    )
                    cs_matrix[i, j] = cos_sim.item()  # Store as scalar
                else:
                    cs_matrix[i, j] = 0  # Set diagonal to 0
        
        # Compute maximum cosine similarity for each client
        maxcs = torch.max(cs_matrix, dim=1)[0] + 1e-5

        # Adjust cosine similarity based on maximum values
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs_matrix[i][j] = cs_matrix[i][j] * maxcs[i]/maxcs[j]
        
        # Compute weight vector
        wv = 1 - torch.max(cs_matrix, dim=1)[0]
        wv = torch.clamp(wv, 0, 1)
        wv = wv / (torch.max(wv) + 1e-10)  # Normalize
        wv[wv == 1] = 0.99  # Avoid division by zero

        # Apply logit function with confidence
        wv = self.confidence * (torch.log((wv/(1-wv)) + 1e-5) + 0.5)
        wv = torch.clamp(wv, 0, 1)

        # Normalize weights
        wv = wv / (torch.sum(wv) + 1e-10)

        return wv
