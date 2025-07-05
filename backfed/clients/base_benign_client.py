"""
Base benign client implementation for FL.
"""

import torch
import time

from typing import Tuple, Dict, Any
from backfed.const import StateDict, Metrics
from backfed.clients.base_client import BaseClient
from backfed.utils import log, test_classifier, test_lstm_reddit, repackage_hidden
from logging import INFO

class BenignClient(BaseClient):
    """
    Base class for all FL clients.
    """
    
    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        client_type: str = "base_benign",
        **kwargs
    ):
        """
        Initialize the client.
        
        Args:
            client_id: Unique identifier 
            dataset: Client dataset
            model: Training model
            client_config: Dictionary containing training configuration
        """
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
            state_dict (StateDict): updated model
        """
        if self.client_config.dataset.upper() == "SENTIMENT140":
            return self.train_albert_sentiment(train_package)
        elif self.client_config.dataset.upper() == "REDDIT":
            return self.train_lstm_reddit(train_package)
        else:
            return self.train_img_classifier(train_package)
    
    def train_img_classifier(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
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
            "global_model_params", "server_round"
        ])

        start_time = time.time()

        # Setup training environment 
        self.model.load_state_dict(train_package["global_model_params"])
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
                        
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

                # Backward pass with gradient masking
                scaler.scale(loss).backward()

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * len(labels)
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_total += len(images)

            epoch_loss = running_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total

        train_loss = epoch_loss
        train_acc = epoch_accuracy
        self.training_time = time.time() - start_time

        # Log final results
        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f}")

        state_dict = self.get_model_parameters()

        training_metrics = {
            "train_clean_loss": train_loss,
            "train_clean_acc": train_acc,
        }

        return len(self.train_dataset), state_dict, training_metrics
    
    def train_lstm_reddit(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train LSTM model for next-word prediction.
        
        Args:
            train_package: Training package from server
            
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
        
        # Training loop
        self.model.train()
        for internal_epoch in range(self.client_config.local_epochs):
            running_loss = 0.0
            epoch_total = 0

            # Initialize hidden state as None, will be set based on actual batch size
            hidden = None

            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)

                current_batch_size = data.size(0)

                # Initialize or reinitialize hidden state if batch size changes
                if hidden is None:
                    hidden = self.model.init_hidden(current_batch_size)
                else:
                    # Check if hidden state batch size matches current batch size
                    if isinstance(hidden, tuple):
                        hidden_batch_size = hidden[0].size(1)
                    else:
                        hidden_batch_size = hidden.size(1)

                    if hidden_batch_size != current_batch_size:
                        hidden = self.model.init_hidden(current_batch_size)

                # Starting each batch, we detach the hidden state
                hidden = repackage_hidden(hidden)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output, hidden = self.model(data, hidden)
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item() * targets.numel()
                epoch_total += targets.numel()
            
            epoch_loss = running_loss / epoch_total
            perplexity = torch.exp(torch.tensor(epoch_loss)).item()
        
        train_loss = epoch_loss
        self.train_perplexity = perplexity
        self.training_time = time.time() - start_time
        
        # Log final results
        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Perplexity: {self.train_perplexity:.4f}")
        
        state_dict = self.get_model_parameters()
        
        training_metrics = {
            "train_clean_loss": train_loss,
            "train_perplexity": self.train_perplexity,
        }
        
        return len(self.train_dataset), state_dict, training_metrics
    
    def train_albert_sentiment(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
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

        train_loss = epoch_loss
        train_acc = epoch_accuracy
        self.training_time = time.time() - start_time

        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f}")

        state_dict = self.get_model_parameters()

        training_metrics = {
            "train_clean_loss": train_loss,
            "train_clean_acc": train_acc,
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
        
        if self.client_config.dataset.upper() != "REDDIT":
            eval_loss, eval_acc = test_classifier(dataset=self.client_config.dataset,
                                                model=self.model, 
                                                test_loader=self.val_loader, 
                                                device=self.device, 
                                                normalization=test_package.get("normalization", None)
                                            )

            metrics = {
                "val_clean_loss": eval_loss, 
                "val_clean_acc": eval_acc,
            }
        else:
            eval_loss, eval_perplexity = test_lstm_reddit(model=self.model,
                                                test_loader=self.val_loader,
                                                device=self.device,
                                                normalization=test_package.get("normalization", None)
                                            )
            
            metrics = {
                "val_clean_loss": eval_loss, 
                "val_perplexity": eval_perplexity,
            }

        return len(self.val_dataset), metrics
    
    @staticmethod
    def get_client_type():
        return "benign"
