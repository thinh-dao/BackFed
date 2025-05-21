"""
Text client implementation for FL.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Any
from logging import INFO
from fl_bdbench.utils import log
from fl_bdbench.clients.base_benign_client import BenignClient

class TextBenignClient(BenignClient):
    """
    Text client implementation for FL.
    Handles training and evaluation for text data.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        dataset_indices: List[List[int]],
        model: nn.Module,
        client_config,
        client_type: str = "text_benign",
        verbose: bool = True,
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
        Train the model for a number of epochs. This function delegates to specialized training
        methods based on model type.
        
        Args:
            train_package: Data package received from server to train the model
            
        Returns:
            num_examples (int): number of examples in the training dataset
            state_dict (Dict[str, torch.Tensor]): updated model parameters
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
        
        # Determine model type and delegate to appropriate training method
        is_transformer = hasattr(self.model, 'albert') or hasattr(self.model, 'transformer_encoder')
        is_lstm = hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn')
        
        # Check if this is a language model (next word prediction) task
        is_language_model = self.client_config.task == "next-word-prediction"
        
        if is_transformer:
            return self._train_albert(server_round)
        elif is_lstm and is_language_model:
            return self._train_lstm_language_model(server_round)
        elif is_lstm:
            return self._train_lstm_classifier(server_round)
        else:
            raise ValueError(f"Unsupported model type for TextBenignClient")
    
    def _train_albert(self, server_round: int) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train Albert/Transformer models for text classification.
        
        Args:
            server_round: Current federated learning round
            
        Returns:
            num_examples, state_dict, training_metrics
        """
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
    
    def _train_lstm_classifier(self, server_round: int) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train LSTM models for text classification.
        
        Args:
            server_round: Current federated learning round
            
        Returns:
            num_examples, state_dict, training_metrics
        """
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
                
                # Process tensor inputs for LSTM models
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Initialize hidden state for LSTM
                hidden = self.model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple([h.to(self.device) for h in hidden])
                else:
                    hidden = hidden.to(self.device)
                
                # Forward pass for LSTM
                outputs, _ = self.model(inputs, hidden)
                
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
    
    def _train_lstm_language_model(self, server_round: int) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train LSTM models for next-word prediction (language modeling).
        
        Args:
            server_round: Current federated learning round
            
        Returns:
            num_examples, state_dict, training_metrics
        """
        start_time = time.time()
        scaler = torch.amp.GradScaler(device=self.device)
        
        # Training loop
        self.model.train()
        for internal_epoch in range(self.client_config.local_epochs):
            running_loss = 0.0
            epoch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if isinstance(targets, torch.Tensor) and len(targets) <= 1:  # Skip small batches
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Process tensor inputs for LSTM models
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Initialize hidden state for LSTM
                hidden = self.model.init_hidden(inputs.size(0))
                if isinstance(hidden, tuple):
                    hidden = tuple([h.to(self.device) for h in hidden])
                else:
                    hidden = hidden.to(self.device)
                
                # Forward pass for LSTM language model
                outputs, _ = self.model(inputs, hidden)
                
                # Reshape for language modeling (vocabulary prediction)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()
                
                # Accumulate loss
                running_loss += loss.item() * len(targets)
                epoch_total += len(targets)
            
            epoch_loss = running_loss / epoch_total
            perplexity = torch.exp(torch.tensor(epoch_loss)).item()
            
            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                    f"- Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                    f"Train Perplexity: {perplexity:.4f}")
        
        self.train_loss = epoch_loss
        self.train_perplexity = perplexity
        self.training_time = time.time() - start_time
        
        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Loss: {self.train_loss:.4f} | "
            f"Train Perplexity: {self.train_perplexity:.4f}")
        
        state_dict = self.get_model_parameters()
        
        training_metrics = {
            "train_clean_loss": self.train_loss,
            "train_clean_perplexity": self.train_perplexity,
        }
        
        return len(self.train_dataset), state_dict, training_metrics
    
    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
        """
        Evaluate the model on the test dataset.
        
        Args:
            test_package: Data package received from server to evaluate the model
            
        Returns:
            num_examples (int): number of examples in the test dataset
            evaluation_metrics (Dict[str, float]): evaluation metrics
        """
        # Validate required keys
        self._check_required_keys(test_package, required_keys=[
            "model_params"
        ])
        
        # Setup evaluation environment
        self.model.load_state_dict(test_package["model_params"])
        
        # Determine model type
        is_transformer = hasattr(self.model, 'albert') or hasattr(self.model, 'transformer_encoder')
        is_lstm = hasattr(self.model, 'lstm') or hasattr(self.model, 'rnn')
        
        # Evaluation loop
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:                
                # Process inputs based on model type and input format
                if isinstance(inputs, dict):
                    # Dictionary inputs (typically for transformer models)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass for transformer models
                    outputs = self.model(**inputs)
                    
                    # Extract logits from transformer outputs if needed
                    if isinstance(outputs, dict):
                        outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                else:
                    # Tensor inputs (typically for LSTM models)
                    inputs = inputs.to(self.device)
                    
                    if is_lstm:
                        # Initialize hidden state for LSTM
                        hidden = self.model.init_hidden(inputs.size(0))
                        if isinstance(hidden, tuple):
                            hidden = tuple([h.to(self.device) for h in hidden])
                        else:
                            hidden = hidden.to(self.device)
                        
                        # Forward pass for LSTM
                        outputs, _ = self.model(inputs, hidden)
                    else:
                        # Generic forward pass for other models
                        outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                
                running_loss += loss.item() * len(labels)
                total += len(labels)
        
        self.eval_loss = running_loss / total
        self.eval_accuracy = correct / total
        
        # Log results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) - "
            f"Test Loss: {self.eval_loss:.4f} | "
            f"Test Accuracy: {self.eval_accuracy:.4f}")
        
        evaluation_metrics = {
            "test_loss": self.eval_loss,
            "test_accuracy": self.eval_accuracy,
        }
        
        return total, evaluation_metrics
