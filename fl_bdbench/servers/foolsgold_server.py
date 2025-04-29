"""
FoolsGold server implementation.
Paper: https://www.usenix.org/conference/raid2020/presentation/fung
"""
import numpy as np
import torch
from typing import Dict, List, Tuple
from logging import INFO

from fl_bdbench.servers.base_server import BaseServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import StateDict

class FoolsGoldServer(BaseServer):
    """
    FoolsGold server that uses cosine similarity to detect and defend against sybil attacks.
    """
    def __init__(self, server_config, server_type="foolsgold", eta=0.1, confidence=1):
        super(FoolsGoldServer, self).__init__(server_config, server_type)
        self.eta = eta
        self.confidence = confidence
        self.update_history: Dict[int, np.ndarray] = {}  # client_id -> update_vector
        log(INFO, f"Initialized FoolsGold server with eta={eta}, confidence={confidence}")

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, StateDict]]) -> StateDict:
        """
        Aggregate client updates using FoolsGold algorithm.
        
        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            The global model state dict after aggregation
        """
        if len(client_updates) == 0:
            return False
        # Extract client IDs and their updates
        client_ids = [client_id for client_id, _, _ in client_updates]
        
        # Store global model state
        global_model_state = {name: param.clone() for name, param in self.global_model.state_dict().items()}

        # Update history for each client
        for client_id, _, client_params in client_updates:
            # Convert model updates to flat vector
            update_vector = []
            for name, param in client_params.items():
                diff = param - global_model_state[name]
                update_vector.append(diff.cpu().numpy().flatten())
            
            update_vector = np.concatenate(update_vector)
            
            # Normalize update vector
            norm = np.linalg.norm(update_vector)
            if norm > 1:
                update_vector = update_vector / norm
                
            # Update history
            if client_id not in self.update_history:
                self.update_history[client_id] = update_vector
            else:
                self.update_history[client_id] += update_vector

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
            if name.endswith('num_batches_tracked'):
                continue
            param.add_(weight_accumulator[name] * self.eta)
            
        return True
    
    def _foolsgold(self, selected_clients) -> np.ndarray:
        """
        Compute FoolsGold weights for the selected clients.
        
        Args:
            selected_clients: List of client IDs
        Returns:
            numpy array of weights for each client
        """
        num_clients = len(selected_clients)
        selected_his = []

        for client_id in selected_clients:
            selected_his.append(self.update_history[client_id])

        # Compute cosine similarity matrix
        M = np.array(selected_his).reshape(num_clients, -1)
        norms = np.linalg.norm(M, axis=1)
        dot_products = M @ M.T
        norms_matrix = np.outer(norms, norms)
        cs_matrix = dot_products / norms_matrix
        cs_matrix = cs_matrix - np.eye(num_clients)

        # Compute maximum cosine similarity for each client
        maxcs = np.max(cs_matrix, axis=1) + 1e-5

        # Adjust cosine similarity based on maximum values
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs_matrix[i][j] = cs_matrix[i][j] * maxcs[i]/maxcs[j]

        # Compute weight vector
        wv = 1 - np.max(cs_matrix, axis=1)
        wv = np.clip(wv, 0, 1)
        wv = wv / np.max(wv)  # Normalize
        wv[wv == 1] = 0.99  # Avoid division by zero
        
        # Apply logit function with confidence
        wv = self.confidence * (np.log((wv/(1-wv)) + 1e-5) + 0.5)
        wv = np.clip(wv, 0, 1)

        return wv
