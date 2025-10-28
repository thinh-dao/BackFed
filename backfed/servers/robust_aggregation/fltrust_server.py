"""
Implementation of FLTrust server for federated learning.
"""

import torch
import torch.nn.functional as F
import copy

from .weakdp_server import NormClippingServer
from backfed.servers.fedavg_server import FedAvgServer
from backfed.utils.logging_utils import log

from typing import Dict, List, Tuple
from logging import INFO
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import instantiate

class FLTrustServer(FedAvgServer):
    """
    FLTrust server implementation that uses cosine similarity with trusted data
    to assign trust scores to client updates.
    """

    def __init__(self, 
        server_config, 
        server_type = "fltrust", 
        eta: float = 0.1,
        m: int = 100, # Number of samples in server's root dataset
    ): 
        self.m = m
        self.eta = eta

        super().__init__(server_config, server_type) # Setup datasets and so on
        
        self.global_lr = self.config.client_config.lr
        self.global_epochs = 1 # Follow original paper

        log(INFO, f"Initialized FLTrust server with m={self.m}, global_lr={self.global_lr}, global_epochs={self.global_epochs}, eta={self.eta}")

    def _prepare_dataset(self):
        """We override the _prepare_dataset function to load auxiliary clean data for the defense."""
        super()._prepare_dataset()
                                    
        if self.m > len(self.testset):
            raise ValueError(f"FLTrust: m ({self.m}) is larger than test set size ({len(self.testset)})")

        random_indices = torch.randperm(len(self.testset))[:self.m]
        self.server_root_data = TensorDataset(torch.stack([self.normalization(self.testset[i][0]) for i in random_indices]),
                                                torch.tensor([self.testset[i][1] for i in random_indices]))
        self.server_dataloader = DataLoader(self.server_root_data, 
                                    batch_size=self.config.client_config.batch_size, # Follow client batch size
                                    shuffle=False, 
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory,
                                )

    def _central_update(self):
        """Perform update on the server's root dataset to obtain the central update."""
        ref_model = copy.deepcopy(self.global_model)
        ref_model.to(self.device)
        ref_model.train()

        # Create server optimizer
        server_optimizer = instantiate(self.config.client_config.optimizer, 
                                       params=ref_model.parameters())
        
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(self.global_epochs):
            for data, label in self.server_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                server_optimizer.zero_grad()
                preds = ref_model(data)
                loss = loss_func(preds, label)
                loss.backward()
                server_optimizer.step()

        return self.parameters_dict_to_vector(ref_model.state_dict()) - self.parameters_dict_to_vector(self.global_model.state_dict())
    
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLTrust mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        central_update_vector = self._central_update()
        central_norm = torch.linalg.norm(central_update_vector)

        score_list = []
        client_ids = []
        updates = []
        total_score = 0

        for client_id, _, local_update in client_updates:
            # Calculate cosine similarity and trust score
            local_update_vector = self.parameters_dict_to_vector(local_update)
            client_cos = F.cosine_similarity(central_update_vector, local_update_vector, dim=0)
            client_cos = max(client_cos.item(), 0)  # ReLU
            local_norm = torch.linalg.norm(local_update_vector)

            # log(INFO, f"FLTrust: Client {client_id} cosine similarity: {client_cos}, local norm: {local_norm.item()}, central norm: {central_norm.item()}")

            # Normalize client update to have the same magnitude as central update
            scale = central_norm / local_norm
            NormClippingServer.scale_update_inplace(local_update, scale, self.trainable_names)

            score_list.append(client_cos)
            client_ids.append(client_id)
            updates.append(local_update)
            total_score += client_cos

        # If all scores are 0, return current global model
        if total_score == 0:
            log(INFO, "FLTrust: All trust scores are 0, keeping current model")
            return False

        fltrust_weights = [score/total_score for score in score_list]
        log(INFO, f"FLTrust weights (client_id, weight): {list(zip(client_ids, fltrust_weights))}")

        weight_accumulator = self.weight_accumulator(updates, fltrust_weights)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True
