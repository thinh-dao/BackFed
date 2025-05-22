"""
Text client implementation for FL.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil

from typing import Dict, List, Tuple, Any
from logging import INFO
from backfed.utils import log
from backfed.clients.base_benign_client import BenignClient

class SentimentBenignClient(BenignClient):
    """
    Sentiment140 benign client implementation.
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        dataset_indices: List[List[int]],
        model: nn.Module,
        client_config,
        client_type: str = "sentiment_benign",
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the text client.

        Args:
            client_id: Unique identifier
            dataset: The whole training dataset
            dataset_indices: Data indices for all clients
            model: Training model
            client_config: Dictionary containing training configuration
            client_type: String for client type identification
            verbose: Whether to print verbose logs
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            dataset_indices=dataset_indices,
            model=model,
            client_config=client_config,
            client_type=client_type,
            verbose=verbose,
            **kwargs
        )

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train Albert/Transformer models for text classification.

        Args:
            server_round: Current federated learning round

        Returns:
            num_examples, state_dict, training_metrics
        """
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "global_model_params", "server_round"
        ])

        # Setup training environment
        self.model.load_state_dict(train_package["global_model_params"])
        server_round = train_package["server_round"]

        start_time = time.time()
        scaler = torch.amp.GradScaler(device=self.device)

        # Training loop
        self.model.train()
        for internal_epoch in range(self.client_config.local_epochs):
            running_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                if isinstance(labels, torch.Tensor) and len(labels) <= 1:  # Skip small batches
                    continue

                # Zero gradients
                self.optimizer.zero_grad()

                # Process dictionary inputs for transformer models
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                # Forward pass for transformer models
                outputs = self.model(**inputs)

                # Extract logits from transformer outputs if needed
                if isinstance(outputs, dict):
                    outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Backward pass
                scaler.scale(loss).backward()

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()

                # Accumulate loss
                running_loss += loss.item() * len(labels)

                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                epoch_correct += (predictions == labels).sum().item()

                epoch_total += len(labels)

            epoch_loss = running_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total

            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                    f"- Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                    f"Train Accuracy: {epoch_accuracy:.4f}")

        self.train_loss = epoch_loss
        self.train_accuracy = epoch_accuracy
        self.training_time = time.time() - start_time

        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Loss: {self.train_loss:.4f} | "
            f"Train Accuracy: {self.train_accuracy:.4f}")

        state_dict = self.get_model_parameters()

        training_metrics = {
            "train_clean_loss": self.train_loss,
            "train_clean_acc": self.train_accuracy,
        }

        return len(self.train_dataset), state_dict, training_metrics
