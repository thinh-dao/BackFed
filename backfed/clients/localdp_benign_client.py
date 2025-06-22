"""
LocalDP client implementation.
"""
import torch
import time

from logging import INFO
from backfed.utils import log
from backfed.clients.base_benign_client import BenignClient
from backfed.const import StateDict, Metrics
from typing import Dict, Any, Tuple

class LocalDPClient(BenignClient):
    
    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        client_type: str = "localDP",
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            client_config=client_config,
            client_type=client_type,
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
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "global_model_params", "server_round", "std_dev", "clipping_norm"
        ])

        start_time = time.time()

        # Setup training environment 
        self.model.load_state_dict(train_package["global_model_params"])
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
        std_dev = train_package["std_dev"]
        clipping_norm = train_package["clipping_norm"]
                        
        # Initialize training tools
        scaler = torch.amp.GradScaler(device=self.device)

        # Training loop
        self.model.train()
        for internal_epoch in range(self.client_config.local_epochs):
            running_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if len(labels) <= 1:  # Skip small batches
                    continue

                # Zero gradients
                self.optimizer.zero_grad()

                # Prepare batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                if normalization:
                    images = normalization(images)

                # Forward pass and loss computation
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                scaler.scale(loss).backward()
                
                # Apply differential privacy to gradients before optimizer step
                self._apply_dp_to_gradients(clipping_norm, std_dev)

                # Optimizer step with privatized gradients
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * len(labels)
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_total += len(images)

            epoch_loss = running_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total

            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                    f"- Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                    f"Train Accuracy: {epoch_accuracy:.4f}")
            
        self.train_loss = epoch_loss
        self.train_accuracy = epoch_accuracy
        self.training_time = time.time() - start_time

        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Loss: {self.train_loss:.4f} | "
            f"Train Accuracy: {self.train_accuracy:.4f}")

        state_dict = self.get_model_parameters()

        training_metrics = {
            "train_clean_loss": self.train_loss,
            "train_clean_acc": self.train_accuracy,
        }

        return len(self.train_dataset), state_dict, training_metrics
    
    def _apply_dp_to_gradients(self, clipping_norm: float, std_dev: float):
        """Apply differential privacy (clipping + noise) to model gradients before optimizer step"""
        # Step 1: Clip gradients
        self._clip_gradients_inplace(clipping_norm)
        # Step 2: Add Gaussian noise to gradients
        self._add_noise_to_gradients_inplace(std_dev)
    
    def _clip_gradients_inplace(self, clipping_threshold: float):
        """Clip gradients to have bounded L2 norm"""
        # Calculate global gradient norm
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Apply clipping if necessary
        if total_norm > clipping_threshold:
            clip_coef = clipping_threshold / total_norm
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def _add_noise_to_gradients_inplace(self, sigma: float):
        """Add Gaussian noise to gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, sigma, size=param.grad.shape, device=param.grad.device)
                param.grad.data.add_(noise)
