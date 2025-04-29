"""
FLDetector server implementation.
"""

import numpy as np
import copy

from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from torch import nn
from fl_bdbench.servers.fedavg_server import UnweightedFedAvgServer
from fl_bdbench.utils import log
from logging import INFO, WARNING

class FLDetectorServer(UnweightedFedAvgServer):
    """FLDetector server implementation."""
    
    def __init__(self, model: nn.Module, window_size: int = 10, **kwargs):
        super().__init__(model, **kwargs)
        self.exclude_list = []
        self.start_round = self.current_round # Start round is the round when the server starts to detect anomalies
        self.init_model = None
        self.window_size = window_size
        
        # Initialize tracking variables
        self.weight_record = []
        self.grad_record = []
        self.malicious_score = np.zeros((1, kwargs.get("num_clients", 100)))
        self.grad_list = []
        self.old_grad_list = []
        self.last_weight = 0
        self.last_grad = 0

    def LBFGS(self, S_k_list: List[np.ndarray], Y_k_list: List[np.ndarray], v: np.ndarray) -> np.ndarray:
        """Implement L-BFGS algorithm for Hessian-vector product approximation."""
        curr_S_k = np.concatenate(S_k_list, axis=1)
        curr_Y_k = np.concatenate(Y_k_list, axis=1)
        S_k_time_Y_k = np.matmul(curr_S_k.T, curr_Y_k)
        S_k_time_S_k = np.matmul(curr_S_k.T, curr_S_k)

        R_k = np.triu(S_k_time_Y_k)
        L_k = S_k_time_Y_k - R_k
        sigma_k = np.matmul(Y_k_list[-1].T, S_k_list[-1]) / (np.matmul(S_k_list[-1].T, S_k_list[-1]))
        D_k_diag = np.diag(S_k_time_Y_k)
        
        upper_mat = np.concatenate([sigma_k * S_k_time_S_k, L_k], axis=1)
        lower_mat = np.concatenate([L_k.T, -np.diag(D_k_diag)], axis=1)
        mat = np.concatenate([upper_mat, lower_mat], axis=0)
        mat_inv = np.linalg.inv(mat)

        approx_prod = sigma_k * v
        p_mat = np.concatenate([np.matmul(curr_S_k.T, sigma_k * v), np.matmul(curr_Y_k.T, v)], axis=0)
        approx_prod -= np.matmul(np.matmul(np.concatenate([sigma_k * curr_S_k, curr_Y_k], axis=1), mat_inv), p_mat)

        return approx_prod

    def simple_mean(self, old_gradients: List[np.ndarray], param_list: List[np.ndarray], 
                   num_malicious: int = 0, hvp: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean of parameters and distances if HVP is provided."""
        if hvp is not None:
            pred_grad = []
            for grad in old_gradients:
                pred_grad.append(grad + hvp)

            pred = np.zeros(len(param_list))
            pred[:num_malicious] = 1
            distance = np.linalg.norm((np.concatenate(pred_grad, axis=1) - np.concatenate(param_list, axis=1)), axis=0)
            distance = distance / np.sum(distance)
        else:
            distance = None

        mean = np.mean(np.concatenate(param_list, axis=1), axis=-1, keepdims=True)
        return mean, distance

    def gap_statistics(self, data: np.ndarray, num_sampling: int, K_max: int, n: int) -> int:
        """Implement gap statistics for optimal cluster number selection."""
        data = np.reshape(data, (data.shape[0], -1))
        data_c = np.ndarray(shape=data.shape)
        
        # Linear transformation
        for i in range(data.shape[1]):
            data_c[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i]) - np.min(data[:,i]))
            
        gaps = []
        s_values = []
        
        for k in range(1, K_max + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++').fit(data_c)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate within-cluster dispersion
            wk = 0
            for i in range(k):
                cluster_points = data_c[labels == i]
                wk += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1))
                
            # Calculate expected dispersion for random data
            wkbs = []
            for _ in range(num_sampling):
                random_data = np.random.uniform(0, 1, size=(n, data.shape[1]))
                kmeans_b = KMeans(n_clusters=k, init='k-means++').fit(random_data)
                wkb = 0
                for i in range(k):
                    cluster_points = random_data[kmeans_b.labels_ == i]
                    wkb += np.sum(np.linalg.norm(cluster_points - kmeans_b.cluster_centers_[i], axis=1))
                wkbs.append(np.log(wkb))
                
            # Calculate gap statistic
            gap = np.mean(wkbs) - np.log(wk)
            sd = np.std(wkbs) * np.sqrt(1 + 1/num_sampling)
            
            gaps.append(gap)
            s_values.append(sd)
            
        # Find optimal number of clusters
        for k in range(1, K_max):
            if gaps[k-1] >= gaps[k] - s_values[k]:
                return k
        return K_max

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> Dict:
        """Aggregate client updates using FLDetector."""
        if len(client_updates) == 0:
            return False

        if self.current_round <= self.start_round:
            self.init_model = {name: param.clone() for name, param in self.global_model.state_dict().items()}

        # Process updates and detect anomalies
        for client_id, _, update in client_updates:
            if client_id in self.exclude_list:
                log(INFO, f"FLDetector: Skipping client {client_id}")
                continue
                
            local_model = copy.deepcopy(self.global_model)
            for name, param in update.items():
                local_model.state_dict()[name].copy_(param)
            
            # Convert model parameters to numpy arrays for detection
            grad_params = [param.detach().cpu().numpy() for name, param in local_model.state_dict().items()]
            self.grad_list.append(grad_params)

        param_list = [np.concatenate([p.reshape(-1, 1) for p in params], axis=0) for params in self.grad_list]
        
        # Get current global weights
        current_weights = [param.detach().cpu().numpy() for name, param in self.global_model.named_parameters()]
        weight = np.concatenate([w.reshape(-1, 1) for w in current_weights], axis=0)

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
            self.malicious_score = np.row_stack((self.malicious_score, distance))

        # Detect anomalies using gap statistics
        if self.malicious_score.shape[0] > self.window_size:
            score = np.sum(self.malicious_score[-self.window_size:], axis=0)
            
            if self.gap_statistics(score, num_sampling=20, K_max=10, 
                                 n=len(client_updates)-len(self.exclude_list)) >= 2:
                
                # Cluster clients into benign and malicious
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(score.reshape(-1, 1))
                labels = kmeans.labels_

                # Identify malicious clients
                if np.mean(score[labels==0]) < np.mean(score[labels==1]):
                    labels = 1 - labels

                log(WARNING, f'FLDetector: Malicious score - Benign: {np.mean(score[labels==1])}, Malicious: {np.mean(score[labels==0])}')
                
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
                self.malicious_score = np.zeros((1, len(client_updates) - len(self.exclude_list)))
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
        
        # Aggregate updates from non-excluded clients
        filtered_updates = [(cid, num, update) for cid, num, update in client_updates 
                          if cid not in self.exclude_list]
        
        return super().aggregate_client_updates(filtered_updates)