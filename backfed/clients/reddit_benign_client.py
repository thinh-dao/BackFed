"""
Text client implementation for FL.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Any
from logging import INFO
from backfed.utils import log, get_batches, repackage_hidden
from backfed.clients.base_benign_client import BenignClient

class RedditBenignClient(BenignClient):
    """
    Reddit benign client implementation for FL.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        dataset_indices: List[List[int]],
        model: nn.Module,
        client_config,
        client_type: str = "reddit_benign",
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
        
    def _set_dataloader(self, dataset, indices):
        """
        Set up train and validation data loaders for the client.
        """
        self.train_dataset = dataset.get_data(self.client_id)
        self.train_loader = get_batches(data_source=self.train_dataset, 
                                        batch_size=self.client_config["batch_size"], 
                                        sequence_length=self.client_config["sequence_length"])
            
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
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
            
            # Get a new batch of data
            hidden = self.model.init_hidden(self.client_config["batch_size"])
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Starting each batch, we detach the hidden state
                hidden = repackage_hidden(hidden)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
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
            "train_perplexity": self.train_perplexity,
        }
        
        return len(self.train_dataset), state_dict, training_metrics
    