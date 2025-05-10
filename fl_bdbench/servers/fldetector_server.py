"""
FLDetector server implementation.
"""

import torch
import numpy as np
import copy

from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from fl_bdbench.servers.defense_categories import AnomalyDetectionServer
from fl_bdbench.utils import log
from logging import INFO, WARNING

class FLDetectorServer(AnomalyDetectionServer):
    """FLDetector server implementation with PyTorch optimizations."""

    def __init__(self, server_config, server_type="fldetector", window_size: int = 10, eta: float = 0.1):
        super().__init__(server_config, server_type, eta)
        self.exclude_list = []
        self.start_round = self.current_round
        self.init_model = None
        self.window_size = window_size

        # Initialize tracking variables as PyTorch tensors
        self.weight_record = []
        self.grad_record = []
        self.malicious_score = torch.zeros(1, self.config.num_clients, device=self.device)
        self.grad_list = []
        self.old_grad_list = []
        self.last_weight = 0
        self.last_grad = 0

    def LBFGS(self, S_k_list: List[torch.Tensor], Y_k_list: List[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """Implement L-BFGS algorithm for Hessian-vector product approximation using PyTorch."""
        # Concatenate tensors along dimension 1
        curr_S_k = torch.cat(S_k_list, dim=1)
        curr_Y_k = torch.cat(Y_k_list, dim=1)
        
        # Matrix multiplications using torch.matmul
        S_k_time_Y_k = torch.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = torch.matmul(curr_S_k.T, curr_S_k)

        # Upper triangular part
        R_k = torch.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - R_k
        
        # Scalar computation
        sigma_k = torch.matmul(Y_k_list[-1].T, S_k_list[-1]) / torch.matmul(S_k_list[-1].T, S_k_list[-1])
        D_k_diag = torch.diag(S_k_time_Y_k)

        # Construct matrix for inversion
        upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
        lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)
        mat = torch.cat([upper_mat, lower_mat], dim=0)
        
        # Matrix inversion
        mat_inv = torch.linalg.inv(mat)

        # Final computation
        approx_prod = sigma_k * v
        p_mat = torch.cat([
            torch.matmul(curr_S_k.T, sigma_k * v), 
            torch.matmul(curr_Y_k.T, v)
        ], dim=0)
        
        approx_prod -= torch.matmul(
            torch.matmul(torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), 
            p_mat
        )

        return approx_prod

    def simple_mean(self, old_gradients: List[torch.Tensor], param_list: List[torch.Tensor],
                   num_malicious: int = 0, hvp: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean of parameters and distances if HVP is provided using PyTorch."""
        # Stack parameters for efficient computation
        stacked_params = torch.cat(param_list, dim=1)
        
        if hvp is not None:
            # Compute predicted gradients
            pred_grad = [grad + hvp for grad in old_gradients]
            stacked_pred_grad = torch.cat(pred_grad, dim=1)
            
            # Create prediction tensor
            pred = torch.zeros(len(param_list), device=self.device)
            if num_malicious > 0:
                pred[:num_malicious] = 1
                
            # Compute distances efficiently
            distance = torch.linalg.norm(stacked_pred_grad - stacked_params, dim=0)
            distance = distance / torch.sum(distance)
        else:
            distance = None

        # Compute mean efficiently
        mean = torch.mean(stacked_params, dim=1, keepdim=True)
        return mean, distance
    
    def detect_anomalies(self, client_updates: List[Tuple[int, int, Dict]]) -> Tuple[List[int], List[int]]:
        """Detect anomalies in the client updates using PyTorch optimizations."""
        if self.current_round <= self.start_round:
            self.init_model = {name: param.clone() for name, param in self.global_model.state_dict().items()}

        # Process updates and detect anomalies
        for client_id, _, update in client_updates:
            if client_id in self.exclude_list:
                log(INFO, f"FLDetector: Skipping client {client_id}")
                continue

            # Apply update to a copy of the global model
            local_model = copy.deepcopy(self.global_model)
            for name, param in update.items():
                local_model.state_dict()[name].copy_(param)

            # Extract model parameters as tensors (keeping on device)
            grad_params = [param.detach() for name, param in local_model.state_dict().items()]
            self.grad_list.append(grad_params)

        # Flatten and concatenate parameters
        param_list = [torch.concat([p.reshape(-1, 1) for p in params], dim=0) for params in self.grad_list]

        # Get current global weights (keeping on device)
        current_weights = [param.detach() for name, param in self.global_model.named_parameters()]
        weight = torch.concat([w.reshape(-1, 1) for w in current_weights], dim=0)

        # Calculate HVP if enough rounds have passed
        hvp = None
        if self.current_round - self.start_round > self.window_size:
            hvp = self.LBFGS(self.weight_record, self.grad_record, weight - self.last_weight)

        # Calculate mean and distances
        grad, distance = self.simple_mean(
            self.old_grad_list,
            param_list,
            len(self.exclude_list),
            hvp
        )

        # Update malicious scores
        if distance is not None and self.current_round - self.start_round > self.window_size:
            self.malicious_score = torch.cat([self.malicious_score, distance.unsqueeze(0)], dim=0)

        # Detect anomalies using gap statistics
        if self.malicious_score.shape[0] > self.window_size:
            score = torch.sum(self.malicious_score[-self.window_size:], dim=0)

            # Convert to numpy for gap statistics
            score_np = score.cpu().numpy()
            
            if self.gap_statistics(score, num_sampling=20, K_max=10,
                                 n=len(client_updates)-len(self.exclude_list)) >= 2:

                # Cluster clients into benign and malicious
                kmeans = KMeans(n_clusters=2, init='k-means++', random_state=self.config.seed)
                kmeans.fit(score_np.reshape(-1, 1))
                labels = kmeans.labels_

                # Identify malicious clients
                if np.mean(score_np[labels==0]) < np.mean(score_np[labels==1]):
                    labels = 1 - labels

                log(WARNING, f'FLDetector: Malicious score - Benign: {np.mean(score_np[labels==1])}, Malicious: {np.mean(score_np[labels==0])}')

                # Update exclude list
                for i, label in enumerate(labels):
                    if label == 0:
                        self.exclude_list.append(i)

                log(WARNING, f"FLDetector: Outliers detected! Restarting from round {self.current_round}")

                # Reset model and tracking variables
                self.global_model.load_state_dict(self.init_model.state_dict())
                self.start_round = self.current_round
                self.weight_record = []
                self.grad_record = []
                self.malicious_score = torch.zeros(1, len(client_updates) - len(self.exclude_list), device=self.device)
                self.grad_list = []
                self.old_grad_list = []
                self.last_weight = 0
                self.last_grad = 0

                return self.global_model.state_dict()

        # Update tracking variables
        self.weight_record.append(weight - self.last_weight)
        self.grad_record.append(grad - self.last_grad)
        if len(self.weight_record) > self.window_size:
            self.weight_record.pop(0)
            self.grad_record.pop(0)

        self.last_weight = weight
        self.last_grad = grad
        self.old_grad_list = param_list
        self.grad_list = []

        client_ids = [client_id for client_id, _, _ in client_updates]
        benign_clients = [client_id for client_id in client_ids if client_id not in self.exclude_list]
        malicious_clients = [client_id for client_id in client_ids if client_id in self.exclude_list]

        return benign_clients, malicious_clients
