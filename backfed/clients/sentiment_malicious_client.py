"""
Text client implementation for FL.
"""
import torch
import torch.nn as nn
import time

from torch.utils.data import Subset, DataLoader
from typing import List, Optional, Dict, Any, Tuple
from logging import INFO
from backfed.utils import log
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.context_actor import ContextActor
from backfed.poisons.text_poison import SentimentPoison

class SentimentMaliciousClient(MaliciousClient):
    """
    Sentiment140 malicious client implementation.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        dataset_indices: List[List[int]],
        model: nn.Module,
        client_config,
        atk_config,
        poison_module: SentimentPoison,
        context_actor: Optional[ContextActor],
        client_type: str = "sentiment140_malicious",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            dataset_indices=dataset_indices,
            model=model,
            client_config=client_config,
            atk_config=atk_config,
            poison_module=poison_module,
            context_actor=context_actor,
            client_type=client_type,
            verbose=verbose,
            **kwargs
        )
        
    def _set_dataloader(self, dataset, indices):
        """
        Set up train and validation data loaders for the client.
        """
        if self.client_config.val_split > 0.0:
            raise ValueError("Validation split is not supported for Sentiment140 yet!")
        
        self.train_dataset = Subset(dataset, indices)
        self.train_dataset = self.poison_module.poison_dataset(self.train_dataset, poisoning_prob=self.atk_config["poison_rate"]) # Modify in-place
        
        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) - Poisoned {len(self.train_dataset)} samples")
            
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)

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
    