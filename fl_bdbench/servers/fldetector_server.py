"""
FLDetector server implementation.
"""

import torch
import numpy as np
import wandb
import os
import glob

from fl_bdbench.utils import save_model_to_wandb_artifact, get_model
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from fl_bdbench.servers.defense_categories import AnomalyDetectionServer, MaliciousClientsIds, BenignClientsIds
from fl_bdbench.utils import log
from logging import INFO, WARNING

class FLDetectorServer(AnomalyDetectionServer):
    """FLDetector server implementation with PyTorch optimizations."""

    def __init__(self, server_config, server_type="fldetector", window_size: int = 10, eta: float = 0.1):
        super().__init__(server_config, server_type, eta)
        self.start_round = self.current_round
        self.window_size = window_size
        self.init_model = None

    def _initialize_model(self):
        """
        Get the initial model.
        """
        if self.config.checkpoint:
            checkpoint = self._load_checkpoint()
            self.global_model = get_model(model_name=self.config.model, num_classes=self.config.num_classes, dataset_name=self.config.dataset)
            self.global_model.load_state_dict(checkpoint['model_state'], strict=True)
            self.start_round = checkpoint['server_round'] + 1

            condition = all(key in checkpoint for key in ['exclude_list', 'weight_record', 'grad_record', 'malicious_score', 'grad_list', 'old_grad_list', 'last_weight', 'last_grad'])
            if condition:
                log(INFO, "FLDetector: Checkpoint contains tracking variables. Loading...")
                self.exclude_list = checkpoint['exclude_list']
                self.weight_record = checkpoint['weight_record']
                self.grad_record = checkpoint['grad_record']
                self.malicious_score = checkpoint['malicious_score']
                self.grad_list = checkpoint['grad_list']
                self.old_grad_list = checkpoint['old_grad_list']
                self.last_weight = checkpoint['last_weight']
                self.last_grad = checkpoint['last_grad']
                self.client_manager.update_rounds_selection(self.exclude_list, start_round=self.start_round)
            else:
                log(WARNING, "FLDetector: Checkpoint does not contain tracking variables.")
                self.exclude_list = []
                self.weight_record = []
                self.grad_record = []
                self.malicious_score = torch.zeros(1, self.config.num_clients, device=self.device)
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

    def _load_checkpoint(self):
        """
        Three ways to load checkpoint:
        1. From W&B
        2. From a specific round
        3. From local path
        """
        if self.config.checkpoint == "wandb":
            # Fetch the model from W&B
            api = wandb.Api()
            artifact = api.artifact(f"{self.config.wandb.entity}/{self.config.wandb.project}/{self.config.dataset}_{self.config.model}:latest")
            local_path = artifact.download()
            log(INFO, f"{self.config.model} checkpoint from W&B is downloaded to: {local_path}")
            resume_model_dict = torch.load(os.path.join(local_path, "model.pth"))
        
        elif isinstance(self.config.checkpoint, int): # Load from specific round
            # Load from checkpoint
            save_dir = os.path.join(os.getcwd(), "checkpoints", f"{self.config.dataset.upper()}_{self.config.aggregator}")
            if self.config.partitioner == "uniform":
                model_path = f"{self.config.model}_round_{self.config.checkpoint}_uniform.pth"
            else:
                # Look for the model with the correct round_number and alpha. If correct alpha is not found, take the model with the highest alpha.
                model_path = os.path.join(save_dir, f"{self.config.model}_round_{self.config.checkpoint}_dir_{self.config.alpha}.pth")
                if not os.path.exists(model_path):
                    model_path_pattern = os.path.join(save_dir, f"{self.config.model}_round_{self.config.checkpoint}_dir_*.pth")
                    model_paths = glob.glob(model_path_pattern)
                    if len(model_paths) == 0:
                        raise FileNotFoundError(f"No checkpoint found for {self.config.model} at round {self.config.checkpoint} with any alpha in {save_dir}")
                    model_path = max(model_paths, key=lambda p: float(p.split('_')[-1].replace('.pth', '')))
                    highest_alpha = float(model_path.split('_')[-1].replace('.pth', ''))
                    log(WARNING, f"No checkpoint found for alpha {self.config.alpha} at round {self.config.checkpoint}. Loading model with highest alpha: {highest_alpha}")

            save_path = os.path.join(save_dir, model_path)
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"No checkpoint found for {self.config.model} at round {self.config.checkpoint} in {save_dir}")
            
            resume_model_dict = torch.load(save_path)
            save_paths = glob.glob(os.path.join(save_dir, f"{self.config.model}_round_{self.config.checkpoint}*.pth"))
            if not save_paths:
                raise FileNotFoundError(f"No checkpoint found for {self.config.model} at round {self.config.checkpoint} in {save_dir}")
            save_path = save_paths[0]  # Assuming we take the first match if multiple files are found
            resume_model_dict = torch.load(save_path)

        else: # Load from local path
            if not os.path.exists(self.config.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found at {self.config.checkpoint}")
            resume_model_dict = torch.load(self.config.checkpoint)
    
        # Update current round
        start_round = resume_model_dict['server_round']
        log(INFO, f"Loaded checkpoint from round {start_round} with metrics: {resume_model_dict['metrics']}")

        return resume_model_dict
    
    def _save_checkpoint(self, model_filename, server_metrics):
        if self.config.save_checkpoint:
            if self.config.partitioner == "dirichlet":
                model_filename = f"{self.config.model}_round_{self.current_round}_dir_{self.config.alpha}.pth"
            else:
                model_filename = f"{self.config.model}_round_{self.current_round}_uniform.pth"
        
            save_dir = os.path.join(os.getcwd(), "checkpoints", f"{self.config.dataset.upper()}_{self.config.aggregator}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, model_filename)
            # Create a dictionary with metrics, model state, server_round, and model_name
            save_dict = {
                'metrics': self.best_metrics,
                'model_state': self.best_model_state,
                'server_round': self.current_round,
                'model_name': self.config.model,

                # Tracking variables
                'exclude_list': self.exclude_list,
                'weight_record': self.weight_record,
                'grad_record': self.grad_record,
                'malicious_score': self.malicious_score,
                'grad_list': self.grad_list,
                'old_grad_list': self.old_grad_list,
                'last_weight': self.last_weight,
                'last_grad': self.last_grad,
            }
            # Save the dictionary
            torch.save(save_dict, save_path)
            log(INFO, f"Checkpoint saved at round {self.current_round} with {self.best_metrics['test_clean_acc'] * 100:.2f}% test accuracy.")

        if self.config.save_model:
            save_dir = os.path.join(self.config.output_dir, "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, model_filename)
            torch.save(self.best_model_state, save_path) # include only model state
            log(INFO, f"Best model saved at round {self.current_round} with {self.best_metrics['test_clean_acc'] * 100:.2f}% test accuracy.")

        if self.config.save_logging in ["wandb", "both"] \
            and self.config.wandb.save_model == True \
            and self.current_round == self.config.wandb.save_model_round:
            save_model_to_wandb_artifact(self.config, self.best_model_state, self.current_round, server_metrics)

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

    def evaluate_detection(self, malicious_clients: List[int], true_malicious_clients: List[int], total_updates: int):
        """
        Evaluate detection performance by comparing detected anomalies with ground truth.
        For FLDetector, we compare exclude_list with all malicious clients.

        Args:
            malicious_clients: List of indices that were detected as anomalous
            true_malicious_clients: List of indices that are actually malicious (ground truth)
            total_updates: Total number of updates being evaluated 

        Returns:
            Dictionary with precision, recall, F1, and FPR for this round.
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
        precision = tp / max(tp + fp, 1)  # Precision = TP / (TP + FP)
        recall = tp / max(tp + fn, 1)     # Recall = TP / (TP + FN)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)  # F1 score
        fpr = fp / max(fp + tn, 1)        # False Positive Rate = FP / (FP + TN)
        acc = (tp + tn) / total_clients

        detection_metrics = {
            "DACC": acc, # Detection accuracy
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
        }
                
        log(INFO, detection_metrics)
        log(INFO, f"═══════════════════════════════════════════════")

        if self.config.save_logging in ["wandb", "both"]:
            wandb.log({**detection_metrics}, step=self.current_round)
        elif self.config.save_logging in ["csv", "both"]:
            self.csv_logger.log({**detection_metrics}, step=self.current_round)
        return detection_metrics
    
    def detect_anomalies(self, client_updates: List[Tuple[int, int, Dict]]) -> Tuple[MaliciousClientsIds, BenignClientsIds]:
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

            grad_params = [param.detach().to(self.device) - self.global_model_params[name].detach() for name, param in client_update.items() if "weight" in name or "bias" in name]
            self.grad_list.append(grad_params)
            client_ids.append(client_id)

        # Flatten and concatenate parameters
        param_list = [torch.concat([p.reshape(-1, 1) for p in params], dim=0) for params in self.grad_list]

        # Get current global weights (keeping on device)
        current_weight_vector = torch.concat([param.reshape(-1, 1) for name, param in self.global_model_params.items() if "weight" in name or "bias" in name], dim=0)

        # Calculate HVP if enough rounds have passed
        hvp = None
        if self.current_round - self.start_round > self.window_size:
            log(INFO, "FLDetector: Calculating Hessian-vector product")
            hvp = self.LBFGS(self.weight_record, self.grad_record, current_weight_vector - self.last_weight)

        # Calculate mean and distances
        grad, distance = self.simple_mean(
            self.old_grad_list,
            param_list,
            len(self.exclude_list),
            hvp
        )

        # Update malicious scores
        if distance is not None and self.current_round - self.start_round > self.window_size:
            distance_extend = torch.zeros(self.config.num_clients, device=self.device)
            distance_extend[client_ids] = distance
            self.malicious_score = torch.cat([self.malicious_score, distance_extend.unsqueeze(0)], dim=0)

        # Detect anomalies using gap statistics
        if self.malicious_score.shape[0] > self.window_size:
            score = torch.sum(self.malicious_score[-self.window_size:], dim=0)

            # Convert to numpy for gap statistics
            score_np = score.cpu().numpy()
            
            optimal_k = self.gap_statistics(score, num_sampling=10, K_max=len(client_updates),
                                n=self.config.num_clients-len(self.exclude_list))
            
            log(INFO, f"FLDetector: Optimal number of clusters from gap_statistics: {optimal_k}")
            if optimal_k >= 2:

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
                
                # Update rounds selection
                log(WARNING, f"FLDetector: Update rounds selection to exclude malicious clients" )
                self.client_manager.update_rounds_selection(self.exclude_list, start_round=self.current_round+1) # exclude malicious clients from the next round

                # Reset model and tracking variables
                self.global_model.load_state_dict(self.init_model)
                self.global_model_params = {name: param.detach().clone().to(self.device) for name, param in self.global_model.state_dict().items()}
                self.start_round = self.current_round

                # reset all tracking variables (except exclude_list)
                self.weight_record = []
                self.grad_record = []
                self.malicious_score = torch.zeros((1, 
                    self.config.num_clients - len(self.exclude_list)), device=self.device)
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
        self.old_grad_list = param_list
        self.grad_list = []

        malicious_clients = [client_id for client_id in client_ids if client_id in self.exclude_list]
        benign_clients = [client_id for client_id in client_ids if client_id not in self.exclude_list]

        return malicious_clients, benign_clients
