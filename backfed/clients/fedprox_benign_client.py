"""
FedProx client implementation.
"""
import torch

from logging import INFO
from backfed.utils import log
from backfed.clients.base_benign_client import BenignClient
from backfed.const import ModelUpdate, Metrics
from typing import Dict, Any, Tuple

class FedProxClient(BenignClient):
    
    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        client_type: str = "fedprox",
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            client_config=client_config,
            client_type=client_type,
            verbose=verbose,
            **kwargs
        )
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, ModelUpdate, Metrics]:
        """
        Train the model for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
            
        Returns: 
            num_examples (int): number of examples in the training dataset
            state_dict (ModelUpdate): updated model parameters
            training_metrics (Dict[str, float]): training metrics
        """
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "global_state_dict", "server_round", "proximal_mu"
        ])

        # Setup training environment 
        self.model.load_state_dict(train_package["global_state_dict"])
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
        proximal_mu = train_package["proximal_mu"]

        global_params_tensor = torch.cat([param.view(-1).detach().clone().requires_grad_(False) for name, param in train_package["global_state_dict"].items()
                    if "weight" in name or "bias" in name]).to(self.device)
                        
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
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                proximal_term = self.model_dist(global_params_tensor=global_params_tensor, gradient_calc=True)
                loss += (proximal_mu / 2) * proximal_term

                # Backward pass
                loss.backward()

                # Optimizer step
                self.optimizer.step()

                running_loss += loss.item() * len(labels)
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_total += len(images)

            epoch_loss = running_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total

            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                    f"- Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                    f"Train Accuracy: {epoch_accuracy:.4f}")

        train_loss = epoch_loss
        train_acc = epoch_accuracy

        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f}")

        training_metrics = {
            "train_clean_loss": train_loss,
            "train_clean_acc": train_acc,
        }

        model_updates = self.weight_diff_dict(client_state_dict=self.model.state_dict(), 
                                              global_state_dict=train_package["global_state_dict"]
                                            )
        
        return len(self.train_dataset), model_updates, training_metrics
