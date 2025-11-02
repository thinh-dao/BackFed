"""
FLDetector server implementation.
"""

import torch
import numpy as np
import wandb

from .anomaly_detection_server import AnomalyDetectionServer
from backfed.utils import get_model, log
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from logging import INFO, WARNING

class FLDetectorServer(AnomalyDetectionServer):
    """FLDetector server implementation with PyTorch optimizations."""

    def __init__(self, server_config, server_type="fldetector", window_size: int = 10, eta: float = 0.5):
        super().__init__(server_config, server_type, eta)
        self.start_round = self.current_round
        self.window_size = window_size
        self.init_model = None
        
        # After calling _init_model()
        if len(self.exclude_list) > 0:
            self.client_manager.update_rounds_selection(self.exclude_list, start_round=self.start_round)

    def _init_model(self):
        """
        Get the initial model.
        """
        if self.config.checkpoint:
            checkpoint = self._load_checkpoint()
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)
            self.global_model.load_state_dict(checkpoint['model_state'], strict=True)
            self.start_round = checkpoint['server_round'] + 1

            condition = all(key in checkpoint for key in ['exclude_list', 'weight_record', 'grad_record', 'malicious_scores_dict', 'grad_list', 'old_grad_list', 'last_weight', 'last_grad'])
            if condition:
                log(INFO, "FLDetector: Checkpoint contains tracking variables. Loading...")
                self.exclude_list = checkpoint['exclude_list']
                self.weight_record = checkpoint['weight_record']
                self.grad_record = checkpoint['grad_record']
                self.malicious_scores_dict = checkpoint['malicious_scores_dict']
                self.grad_list = checkpoint['grad_list']
                self.old_grad_list = checkpoint['old_grad_list']
                self.last_weight = checkpoint['last_weight']
                self.last_grad = checkpoint['last_grad']
            else:
                log(WARNING, "FLDetector: Checkpoint does not contain tracking variables.")
                self.exclude_list = []
                self.weight_record = []
                self.grad_record = []
                self.malicious_scores_dict = {}
                self.grad_list = []
                self.old_grad_list = []
                self.last_weight = 0
                self.last_grad = 0
            
        elif self.config.pretrain_model_path != None:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset, pretrain_model_path=self.config.pretrain_model_path)
        
        else:
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)

        self.global_model = self.global_model.to(self.device)

        if self.config.wandb.save_model == True and self.config.wandb.save_model_round == -1:
            self.config.wandb.save_model_round = self.start_round + self.config.num_rounds

    def _get_save_dict(self):
        return {
            'metrics': self.best_metrics,
            'model_state': self.best_model_state,
            'server_round': self.current_round,
            'model_name': self.config.model,

            # Tracking variables
            'exclude_list': self.exclude_list,
            'weight_record': self.weight_record,
            'grad_record': self.grad_record,
            'malicious_scores_dict': self.malicious_scores_dict,
            'grad_list': self.grad_list,
            'old_grad_list': self.old_grad_list,
            'last_weight': self.last_weight,
            'last_grad': self.last_grad,
        }

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
        stacked_params = torch.stack(param_list, dim=1)
        
        if hvp is not None:
            # Compute predicted gradients
            pred_grad = [grad + hvp for grad in old_gradients]
            stacked_pred_grad = torch.stack(pred_grad, dim=1)
                
            # Compute distances efficiently
            distance = torch.linalg.norm(stacked_pred_grad - stacked_params, dim=0)
            distance = distance / torch.sum(distance)
        else:
            distance = None

        # Compute mean efficiently
        mean = torch.mean(stacked_params, dim=1, keepdim=True)
        return mean, distance

    def evaluate_detection(self, benign_clients: List[int], malicious_clients: List[int], true_malicious_clients: List[int], total_updates: int):
        """
        Evaluate detection performance by comparing detected anomalies with ground truth.
        For FLDetector, we compare exclude_list with all malicious clients.

        Args:
            benign_clients: List of indices that were detected as benign
            malicious_clients: List of indices that were detected as anomalous
            true_malicious_clients: List of indices that are actually malicious (ground truth)
            total_updates: Total number of updates being evaluated 

        Returns:
            Dictionary with TPR, TNR, and DACC for this round.
        """
        detected_set = set(self.exclude_list)
        true_set = set(self.client_manager.get_malicious_clients())
        total_clients = self.config.num_clients

        log(INFO, f"═══ {self.__class__.__name__} detection results ═══")
        log(INFO, f"Predicted malicious clients: {list(detected_set)}")
        log(INFO, f"Ground-truth malicious clients: {list(true_set)}")

        # Calculate metrics for this round
        tp = len(detected_set.intersection(true_set))
        fp = len(detected_set - true_set)
        fn = len(true_set - detected_set)
        tn = total_clients - tp - fp - fn

        # Calculate key metrics for this round
        tpr = tp / max(tp + fn, 1)        # TPR = TP / (TP + FN) = Recall
        tnr = tn / max(tn + fp, 1)        # TNR = TN / (TN + FP) = Specificity
        dacc = (tp + tn) / total_clients  # Detection Accuracy

        detection_metrics = {
            "TPR": tpr,
            "TNR": tnr,
            "DACC": dacc,
        }
        
        from backfed.utils import log_detection_metrics
        log_detection_metrics(detection_metrics,
                            true_positives=tp,
                            false_positives=fp,
                            true_negatives=tn,
                            false_negatives=fn)
        log(INFO, f"═══════════════════════════════════════════════")

        if self.config.save_logging in ["wandb", "both"]:
            wandb.log({**detection_metrics}, step=self.current_round)
        elif self.config.save_logging in ["csv", "both"]:
            self.csv_logger.log({**detection_metrics}, step=self.current_round)
        return detection_metrics
    
    def detect_anomalies(self, client_updates: List[Tuple[int, int, Dict]]):
        """Detect anomalies in the client updates using PyTorch optimizations."""
        if self.current_round <= self.start_round:
            self.init_model = {name: param.detach().clone() for name, param in self.global_model.state_dict().items()}

        log(INFO, f"FLDetector: Detected malicious clients at round {self.current_round} - {self.exclude_list}")

        client_ids = []
        # Process updates and detect anomalies
        for client_id, _, client_update in client_updates:
            if client_id in self.exclude_list:
                log(WARNING, f"FLDetector: Skipping client {client_id}")
                continue

            self.grad_list.append(self.parameters_dict_to_vector(client_update))
            client_ids.append(client_id)

        # Get current global weights (keeping on device)
        current_weight_vector = torch.concat([
            param.reshape(-1, 1) 
            for name, param in self.global_model.state_dict().items() 
            if not any(skip_name in name for skip_name in ['running_mean', 'running_var', 'num_batches_tracked'])
        ], dim=0)

        # Calculate HVP if enough rounds have passed
        hvp = None
        if self.current_round - self.start_round > self.window_size:
            log(INFO, "FLDetector: Calculating Hessian-vector product")
            hvp = self.LBFGS(self.weight_record, self.grad_record, current_weight_vector - self.last_weight)

        # Calculate mean and distances
        grad, distance = self.simple_mean(
            self.old_grad_list,
            self.grad_list,
            len(self.exclude_list),
            hvp
        )

        # Update malicious scores
        if distance is not None and self.current_round - self.start_round > self.window_size:
            # Update scores in the dictionary for each client
            for i, client_id in enumerate(client_ids):
                if client_id not in self.malicious_scores_dict:
                    self.malicious_scores_dict[client_id] = []
                
                # Add new score for this round
                if i < len(distance):
                    self.malicious_scores_dict[client_id].append(distance[i].item())
                
                # Keep only the last window_size scores
                if len(self.malicious_scores_dict[client_id]) > self.window_size:
                    self.malicious_scores_dict[client_id] = self.malicious_scores_dict[client_id][-self.window_size:]

        # Detect anomalies using gap statistics
        if self.current_round - self.start_round > self.window_size:
            # Calculate sum of scores for each client over the window
            client_scores = {}
            for client_id, scores in self.malicious_scores_dict.items():
                if len(scores) > 0:  # Only consider clients with scores
                    client_scores[client_id] = sum(scores)
            
            # Convert to numpy array for clustering
            if client_scores:
                client_ids_list = list(client_scores.keys())
                score_np = np.array([client_scores[cid] for cid in client_ids_list]).reshape(-1, 1)
                
                optimal_k = self.gap_statistics(score_np, num_sampling=10, K_max=len(client_updates),
                                    n=len(client_scores))
                
                log(INFO, f"FLDetector: Optimal number of clusters from gap_statistics: {optimal_k}")
                if optimal_k >= 2:

                    # Cluster clients into benign and malicious
                    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=self.config.seed)
                    kmeans.fit(score_np)
                    labels = kmeans.labels_

                    # Identify malicious clients
                    if np.mean(score_np[labels==0]) < np.mean(score_np[labels==1]):
                        labels = 1 - labels

                    log(WARNING, f'FLDetector: Malicious score - Benign: {np.mean(score_np[labels==1])}, Malicious: {np.mean(score_np[labels==0])}')

                    # Update exclude list
                    for i, label in enumerate(labels):
                        if label == 0:
                            self.exclude_list.append(client_ids_list[i])

                    log(WARNING, f"FLDetector: Outliers detected! Restarting from round {self.current_round}")
                    
                    # Update rounds selection
                    log(WARNING, f"FLDetector: Update rounds selection to exclude malicious clients" )
                    self.client_manager.update_rounds_selection(self.exclude_list, start_round=self.current_round+1) # exclude malicious clients from the next round

                    # Reset model and tracking variables
                    self.global_model.load_state_dict(self.init_model)
                    self.start_round = self.current_round

                    # reset all tracking variables (except exclude_list)
                    self.weight_record = []
                    self.grad_record = []
                    self.malicious_score_dict = {}
                    self.grad_list = []
                    self.old_grad_list = []
                    self.last_weight = 0
                    self.last_grad = 0

                    return [], [] # return empty lists to indicate restart

        # Update tracking variables
        self.weight_record.append(current_weight_vector - self.last_weight)
        self.grad_record.append(grad - self.last_grad)
        if len(self.weight_record) > self.window_size:
            self.weight_record.pop(0)
            self.grad_record.pop(0)

        self.last_weight = current_weight_vector
        self.last_grad = grad
        self.old_grad_list = self.grad_list
        self.grad_list = []

        malicious_clients = [client_id for client_id in client_ids if client_id in self.exclude_list]
        benign_clients = [client_id for client_id in client_ids if client_id not in self.exclude_list]

        return malicious_clients, benign_clients
