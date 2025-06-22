"""
Text client implementation for FL.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Any, Optional
from logging import INFO
from backfed.utils import log, repackage_hidden
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.context_actor import ContextActor
from backfed.poisons.text_poison import RedditPoison

class RedditMaliciousClient(MaliciousClient):
    """
    Reddit malicious client implementation for FL.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        model: nn.Module,
        client_config,
        atk_config,
        poison_module: RedditPoison,
        context_actor: Optional[ContextActor],
        client_type: str = "reddit_malicious",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
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
        self.train_dataset = dataset.get_data(self.client_id)
        self.train_loader = self.poison_module.get_poisoned_batches(data_source=self.train_dataset, 
                                        batch_size=self.client_config["batch_size"], 
                                        sequence_length=self.client_config["sequence_length"], poisoning_prob=self.atk_config["poison_rate"])
            
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
                # Compute loss only for the last word in each sequence
                last_output = output[-1]  # shape: (batch_size, vocab_size)
                last_targets = targets[-self.client_config["batch_size"]:]  # shape: (batch_size,)
                loss = self.criterion(last_output, last_targets)
                
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
            f"Train Backdoor Loss: {self.train_loss:.4f} | "
            f"Train Perplexity: {self.train_perplexity:.4f}")
        
        state_dict = self.get_model_parameters()
        
        training_metrics = {
            "train_backdoor_loss": self.train_loss,
            "train_perplexity": self.train_perplexity,
        }
        
        return len(self.train_dataset), state_dict, training_metrics
    