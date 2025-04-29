"""
Base benign client implementation for FL.
"""

import torch
import time

from typing import Tuple, Dict, Any
from fl_bdbench.const import StateDict, Metrics
from fl_bdbench.clients.base_client import BaseClient
from fl_bdbench.utils import model_dist_layer, test
from fl_bdbench.utils import log
from logging import INFO

class BenignClient(BaseClient):
    """
    Base class for all FL clients.
    """
    
    def __init__(self, client_id, dataset, dataset_indices, model, client_config, **kwargs):
        """
        Initialize the client.
        
        Args:
            client_id: Unique identifier 
            dataset: The whole training dataset
            dataset_indices: Data indices for all clients
            model: Training model
            client_config: Dictionary containing training configuration
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            dataset_indices=dataset_indices,
            model=model,
            client_config=client_config,
            client_type="benign",
            **kwargs
        )

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        """
        Train the model for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
            
        Returns: 
            num_examples (int): number of examples in the training dataset
            state_dict (StateDict): updated model parameters
            training_metrics (Dict[str, float]): training metrics
        """
        # Check if the required keys are present in the train_package
        self._check_required_keys(train_package, required_keys=["global_model_params", "server_round"])

        # Load the global model parameters
        self.model.load_state_dict(train_package["global_model_params"])

        # Get normalization function
        normalization = train_package.get("normalization", None)

        self.model.train()
        start_time = time.time()
        
        # Reset GPU memory tracking if using CUDA
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        if train_package.get('proximal_mu', None) is not None:
            proximal_mu = train_package['proximal_mu']
        else:
            proximal_mu = None
    
        self.model.train()
        scaler = torch.amp.GradScaler(device=self.device)
        
        num_epochs = self.client_config.get("local_epochs", 2)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0.0
            epoch_total = 0
            for images, labels in self.train_loader:
                if len(images) == 1:
                    continue
                if normalization:
                    images = normalization(images)
                inputs = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                    
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    if proximal_mu is not None:
                        proximal_term = model_dist_layer(self.model, {k: v.detach().clone().requires_grad_(False) for k, v in train_package["global_model_params"].items()})
                        loss = self.criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
                    else:
                        loss = self.criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                epoch_loss += loss.item() * len(images)
                epoch_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                epoch_total += len(images)
            
            # Epoch metrics
            epoch_loss = epoch_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total
                
        # Calculate average metrics
        self.training_time = time.time() - start_time

        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {train_package['server_round']} - Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.4f}")
        
        state_dict = self.get_model_parameters()

        training_metrics = {
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
        }
        
        return len(self.train_dataset), state_dict, training_metrics
    
    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model on client's validation data.
        
        Args:
            test_package: Data package received from server to evaluate the model (e.g., global model weights, learning rate, etc.)
            
        Returns:
            num_examples (int): number of examples in the test dataset
            evaluation_metrics (Dict[str, float]): evaluation metrics
        """
        if self.val_loader is None:
            raise Exception("There is no validation data for this client")
        
        required_keys = ["global_model_params"]
        for key in required_keys:
            assert key in test_package, f"{key} not found in test_package for benign client"

        # Update model weights and evaluate
        self.model.load_state_dict(test_package["global_model_params"])
        self.model.eval()
        loss, accuracy = test(model=self.model, 
            test_loader=self.val_loader, 
            device=self.device, 
            normalization=test_package.get("normalization", None)
        )

        metrics = {
            "val_clean_loss": loss, 
            "val_clean_acc": accuracy,
        }
        
        return len(self.val_dataset), metrics
    
    @staticmethod
    def get_client_type():
        return "benign"
