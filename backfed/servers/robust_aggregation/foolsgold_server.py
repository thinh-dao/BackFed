"""
FoolsGold server implementation.
Paper: https://www.usenix.org/conference/raid2020/presentation/fung
"""
import torch

from typing import Dict, List, Tuple
from logging import INFO, WARNING
from backfed.servers.fedavg_server import FedAvgServer
from backfed.utils import get_model, log
from backfed.const import ModelUpdate, client_id, num_examples

class FoolsGoldServer(FedAvgServer):
    """
    FoolsGold server that uses cosine similarity to detect and defend against sybil attacks.
    """
    def __init__(self, server_config, server_type="foolsgold", confidence=1, eta=0.1):
        self.confidence = confidence
        self.eta = eta
        self.update_history: Dict[int, torch.Tensor] = {}  # client_id -> update_vector

        super(FoolsGoldServer, self).__init__(server_config, server_type) # Initialize here to avoid overridding update_history

        log(INFO, f"Initialized FoolsGold server with confidence={confidence}, eta={eta}")

    def _init_model(self):
        """
        Get the initial model.
        """
        if self.config.checkpoint:
            checkpoint = self._load_checkpoint()
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)
            self.global_model.load_state_dict(checkpoint['model_state'], strict=True)
            self.start_round = checkpoint['server_round'] + 1

            if 'update_history' in checkpoint:
                log(INFO, "FoolsGold: Checkpoint contains update_history. Loading...")
                self.update_history = checkpoint['update_history']
            else:
                log(WARNING, "FoolsGold: Checkpoint does not contain update_history.")
                self.update_history = {}
            
        elif self.config.pretrain_model_path != None:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset, pretrain_model_path=self.config.pretrain_model_path)
        
        else:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)

        self.global_model = self.global_model.to(self.device)

        if self.config.wandb.save_model == True and self.config.wandb.save_model_round == -1:
            self.config.wandb.save_model_round = self.start_round + self.config.num_rounds

    # We need to also store update_history
    def _get_save_dict(self):
        return {
            'metrics': self.best_metrics,
            'model_state': self.best_model_state,
            'server_round': self.current_round,
            'model_name': self.config.model.lower(),
            'update_history': self.update_history,
        }

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Aggregate client updates using FoolsGold algorithm.

        Args:
            client_updates: List of tuples (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        # Extract client IDs and convert updates to vectors (cache for reuse)
        client_ids = []
        update_vectors = []
        updates = []
        
        for client_id, _, client_update in client_updates:
            updates.append(client_update)
            client_ids.append(client_id)
            update_vector = self.parameters_dict_to_vector(client_update).to(self.device)
            update_vectors.append(update_vector)

        # Update history for each client
        for client_id, update_vector in zip(client_ids, update_vectors):
            # Normalize update vector for history
            v = update_vector / (update_vector.norm() + 1e-12)
            self.update_history[client_id] = self.update_history.get(client_id, 0) + v.detach().cpu()

        # Calculate FoolsGold weights
        foolsgold_weights = self._foolsgold(client_ids).tolist()
        log(INFO, f"FoolsGold weights (client_id, weight): {list(zip(client_ids, foolsgold_weights))}")

        weight_accumulator = self.weight_accumulator(updates, foolsgold_weights)
    
        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

    def _foolsgold(self, selected_clients) -> torch.Tensor:
        """
        Compute FoolsGold weights for the selected clients. 
        Different from official codebase: We add normalization so the sum of weights is 1.
        Args:
            selected_clients: List of client IDs
        
        Returns:
            torch.Tensor of weights for each client
        """
        n_clients = len(selected_clients)
        
        # Stack client update histories [n_clients, feature_dim]
        grads = torch.stack(
            [self.update_history[cid] for cid in selected_clients], dim=0
        ).to(self.device)
        
        # Compute cosine similarity matrix (vectorized)
        # Normalize each row
        norms = torch.linalg.norm(grads, dim=1, keepdim=True)
        normalized_grads = grads / (norms + 1e-10)
        
        # cs = cosine_similarity - eye(n)
        cs = torch.mm(normalized_grads, normalized_grads.T) - torch.eye(n_clients, device=self.device)
        
        # maxcs = max(cs, axis=1)
        maxcs = torch.max(cs, dim=1)[0]
        
        # Pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        
        # wv = 1 - max(cs, axis=1)
        wv = 1 - torch.max(cs, dim=1)[0]
        wv[wv > 1] = 1
        wv[wv < 0] = 0
        
        # Rescale so that max value is 1
        wv = wv / torch.max(wv)
        wv[wv == 1] = 0.99
        
        # Logit function
        wv = torch.log(wv / (1 - wv)) + 0.5
        wv[(torch.isinf(wv) | (wv > 1))] = 1
        wv[wv < 0] = 0
        
        # Normalize weights to sum to 1
        wv /= torch.sum(wv) + 1e-10
        
        return wv
