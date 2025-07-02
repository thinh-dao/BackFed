"""
Text client implementation for FL.
"""
import torch
import torch.nn as nn
import time
import copy

from typing import Optional, Dict, Any, Tuple
from logging import INFO, WARNING
from backfed.utils import log
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.context_actor import ContextActor
from backfed.poisons.text_poison import SentimentPoisonBert

class SentimentMaliciousClient(MaliciousClient):
    """
    Sentiment140 malicious client implementation.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        model: nn.Module,
        client_config,
        atk_config,
        poison_module: SentimentPoisonBert,
        context_actor: Optional[ContextActor],
        client_type: str = "sentiment140_malicious",
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
            "global_model_params", "selected_malicious_clients", "server_round"
        ])

        start_time = time.time()
        
        # Setup training environment
        self.model.load_state_dict(train_package["global_model_params"])
        selected_malicious_clients = train_package["selected_malicious_clients"]
        server_round = train_package["server_round"]
        
        # Verify client is selected for poisoning
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"

        # Initialize poison attack
        self._update_and_sync_poison(selected_malicious_clients, server_round)

        # Setup poisoned dataloader if poison_mode is offline
        if self.atk_config.poison_mode == "offline":
            self.set_poisoned_dataloader()

        # Setup training protocol
        proximal_mu = train_package.get('proximal_mu', None) if self.atk_config.follow_protocol else None

        # Initialize training tools
        scaler = torch.amp.GradScaler(device=self.device)

        if self.atk_config.poisoned_is_projection or proximal_mu is not None:
            global_params_tensor = torch.cat([param.view(-1).detach().clone().requires_grad_(False) for name, param in train_package["global_model_params"].items()
                                  if "weight" in name or "bias" in name]).to(self.device)

        if self.atk_config["step_scheduler"]:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.atk_config["step_size"],
                gamma=0.1
            )

        # Determine number of training epochs
        if self.atk_config.poison_until_convergence:
            num_epochs = 100  # Large number for convergence-based training
            log(WARNING, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                "- Training until convergence of backdoor loss")
        else:
            num_epochs = self.atk_config.poison_epochs

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
                
                # Forward pass and loss computation
                with torch.amp.autocast("cuda"):
                    if self.atk_config.poison_mode == "multi_task":
                        # Handle multi-task poisoning
                        clean_inputs = copy.deepcopy(inputs)
                        clean_labels = labels.detach().clone()
                        poisoned_inputs = self.poison_module.poison_inputs(inputs)
                        poisoned_labels = self.poison_module.poison_labels(labels)

                        # Compute losses for both clean and poisoned data in a single forward pass
                        clean_output = self.model(**inputs)
                        poisoned_output = self.model(**poisoned_inputs)

                        clean_loss = self.criterion(clean_output, clean_labels)
                        poisoned_loss = self.criterion(poisoned_output, poisoned_labels)

                        # Extract logits from transformer outputs if needed
                        if isinstance(outputs, dict):
                            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                            
                        # Combine losses according to attack alpha
                        loss = (self.atk_config.attack_alpha * poisoned_loss +
                               (1 - self.atk_config.attack_alpha) * clean_loss)

                    elif self.atk_config.poison_mode in ["online", "offline"]:
                        if self.atk_config.poison_mode == "online":
                            inputs, labels = self.poison_module.poison_batch(batch=(inputs, labels))

                        # Forward pass and loss computation
                        outputs = self.model(**inputs)
                        
                        # Extract logits from transformer outputs if needed
                        if isinstance(outputs, dict):
                            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                            
                        loss = self.criterion(outputs, labels)

                    else:
                        raise ValueError(
                            f"Invalid poison_mode: {self.atk_config.poison_mode}. "
                            f"Expected one of: ['multi_task', 'online', 'offline']"
                        )
                    
                    # Add proximal term if needed
                    if proximal_mu is not None:
                        proximal_term = self.model_dist(global_params_tensor=global_params_tensor, gradient_calc=True)
                        loss += (proximal_mu / 2) * proximal_term

                # Backward pass
                scaler.scale(loss).backward()

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()

                # Project poisoned model parameters
                if self.atk_config.poisoned_is_projection and \
                    ( (batch_idx + 1) % self.atk_config.poisoned_projection_frequency == 0 or
                     (batch_idx == len(self.train_loader) - 1) ):
                    self._projection(global_params_tensor)

                running_loss += loss.item() * len(labels)
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_total += len(labels)

            epoch_loss = running_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total

            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                    f"- Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                    f"Train Accuracy: {epoch_accuracy:.4f}")
                
            # Check convergence
            if (self.atk_config["poison_until_convergence"] and
                epoch_loss < self.atk_config["poison_convergence_threshold"]):
                break

            # Step scheduler if needed
            if self.atk_config["step_scheduler"]:
                scheduler.step()

        train_loss = epoch_loss
        train_acc = epoch_accuracy
        self.training_time = time.time() - start_time

        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Backdoor Loss: {train_loss:.4f} | "
            f"Train Backdoor Accuracy: {train_acc:.4f}")

        # Prepare return values
        if self.atk_config["scale_weights"]:
            state_dict = self.get_model_replacement_parameters(
                scale_factor=self.atk_config["scale_factor"],
                global_params=train_package["global_model_params"]
            )
        else:
            state_dict = self.get_model_parameters()

        training_metrics = {
            "train_backdoor_loss": train_loss,
            "train_backdoor_acc": train_acc,
        }

        return len(self.train_dataset), state_dict, training_metrics
    