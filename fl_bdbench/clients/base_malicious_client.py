"""
Base malicious client implementation for FL.
"""
import random
import time
import torch
import copy
import ray

from logging import INFO, WARNING
from fl_bdbench.utils import log
from fl_bdbench.poisons import IBA, A3FL, Poison
from fl_bdbench.context_actor import ContextActor
from fl_bdbench.clients.base_benign_client import BenignClient
from fl_bdbench.utils import model_dist_layer
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from torch.nn.utils import vector_to_parameters
from fl_bdbench.const import StateDict, Metrics
from typing import Tuple, Dict, Any

class MaliciousClient(BenignClient):
    """
    Malicious client implementation for FL.
    """ 

    def __init__(self, client_id, dataset, dataset_indices, model, client_config, atk_config, poison_module, context_actor: ContextActor, **kwargs):
        """
        Initialize the malicious client.
        
        Args:
            client_id: Unique identifier 
            dataset: The whole training dataset
            dataset_indices: Data indices for all clients
            model: Training model
            client_config: Dictionary containing training configuration
            atk_config: Dictionary containing attack configuration
            poison_module: Poison module for to inject trigger
            context_actor: Context actor for resource synchronization
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            dataset_indices=dataset_indices,
            model=model,
            client_config=client_config,
            client_type="malicious",
            **kwargs
        )
        self.atk_config = atk_config
        self.atk_config.update(kwargs)
        self.context_actor = context_actor
        self.poison_module = poison_module
        self.poison_module.set_client_id(client_id)
        self.train_backdoor_loss = 0
        self.train_backdoor_acc = 0

    def _set_optimizer(self):
        if self.atk_config.use_atk_optimizer:
            self.optimizer = instantiate(self.atk_config.atk_optimizer, params=self.model.parameters())
        else:
            super()._set_optimizer()

    def _set_poisoned_dataloader(self):
        if self.atk_config.poison_type == "offline":
            poisoned_dataset = PoisonedDataset(self.train_dataset, self.poison_module, self.atk_config.poison_rate)
            self.train_loader = DataLoader(poisoned_dataset, 
                                    batch_size=self.client_config["train_batch_size"], 
                                    shuffle=True, pin_memory=True, 
                                )
        elif self.atk_config.poison_type == "online":
            self.train_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.client_config["train_batch_size"], 
                                    shuffle=True, pin_memory=True, 
                                    collate_fn=train_poison_wrapper(self.poison_module), 
                                )
        elif self.atk_config.poison_type == "multi_task":
            log(INFO, "Multi-task poisoning happens during training, no data poisoning required.")
        else:
            raise ValueError(f"Invalid poison type: {self.atk_config.poison_type}")

    def _update_and_sync_poison(self, selected_malicious_clients, server_round, normalization):
        """
        The first malicious client updates and synchronizes the poison module (Only for IBA and A3FL).
        In serial mode, the instance of poison module is shared by server and all clients, so the poison module is automatically updated.
        In parallel mode, each malicious client has its own instance of poison module, so the poison module needs to be synchronized.
        """

        # Only IBA and A3FL requires resource synchronization
        if type(self.poison_module) not in [IBA, A3FL]:
            return 

        if self.client_id == selected_malicious_clients[0]:
            self.poison_module.poison_warmup(  
                client_id=self.client_id,
                initial_model=self.model, 
                dataloader=self.train_loader, 
                selected_malicious_clients=selected_malicious_clients,
                server_round=server_round,
                normalization=normalization
            )
        
        # Update and synchronize the poison module in parallel mode
        if self.training_mode == "parallel":
            if self.client_id == selected_malicious_clients[0]:
                if isinstance(self.poison_module, IBA):
                    resource_package = {
                        "iba_atk_model": self.poison_module.atk_model.state_dict()
                    }
                elif isinstance(self.poison_module, A3FL):
                    resource_package = {
                        "a3fl_trigger": self.poison_module.trigger_image.detach().clone()
                    }
                self.context_actor.update_resource.remote(client_id=self.client_id, resource_package=resource_package, round_number=server_round)
            else:
                if isinstance(self.poison_module, IBA):
                    resource_package = ray.get(self.context_actor.wait_for_resource.remote(resource_key="iba_atk_model", round_number=server_round))
                    self.poison_module.atk_model.load_state_dict(resource_package["iba_atk_model"])
                elif isinstance(self.poison_module, A3FL):
                    resource_package = ray.get(self.context_actor.wait_for_resource.remote(resource_key="a3fl_trigger", round_number=server_round))
                    self.poison_module.trigger_image = resource_package["a3fl_trigger"]

    def train(self, train_package):
        """Train the neurotoxin malicious client.
        
        Args:
            train_package (dict): Contains training parameters including:
                - poison_module: The poison module to use
                - global_model_params: Global model parameters
                - selected_malicious_clients: List of selected malicious clients
                - server_round: Current server round
                - normalization: Optional normalization function
        
        Returns:
            tuple: (num_examples, client_updates, training_metrics)
                - num_examples (int): number of examples in the training dataset
                - state_dict (StateDict): updated model parameters
                - training_metrics (Dict[str, float]): training metrics
        """
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "poison_module", "global_model_params", "selected_malicious_clients", "server_round"
        ])

        start_time = time.time()
        
        # Setup training environment
        self.poison: Poison = train_package["poison_module"]
        self.model.load_state_dict(train_package["global_model_params"])
        selected_malicious_clients = train_package["selected_malicious_clients"]
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
        
        # Verify client is selected for poisoning
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"
        
        # Initialize poison attack
        self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)
        super()._set_poisoned_dataloader()

        # Setup training protocol
        proximal_mu = self.atk_config.get('proximal_mu', None) if self.atk_config.follow_protocol else None
        
        # Initialize training tools
        scaler = torch.amp.GradScaler(device=self.device)
        
        if self.atk_config.poisoned_is_projection:
            global_params_tensor = torch.cat([param.view(-1) for name, param in train_package["global_model_params"].items() 
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
        for internal_epoch in range(num_epochs):
            running_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if len(labels) <= 1:  # Skip small batches
                    continue
                
                # Prepare batch
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if normalization:
                    images = normalization(images)

                # Forward pass and loss computation
                with torch.amp.autocast("cuda"):
                    if self.atk_config.poison_type == "multi_task":
                        # Handle multi-task poisoning
                        clean_images = images.clone()
                        clean_labels = labels.clone()
                        poisoned_images = self.poison.poison_inputs(clean_images)
                        poisoned_labels = self.poison.poison_labels(clean_labels)

                        # Compute clean and poisoned losses
                        clean_output = self.model(clean_images)
                        clean_loss = self.criterion(clean_output, clean_labels)

                        poisoned_output = self.model(poisoned_images)
                        poisoned_loss = self.criterion(poisoned_output, poisoned_labels)

                        # Combine losses according to attack alpha
                        loss = (self.atk_config.attack_alpha * poisoned_loss + 
                               (1 - self.atk_config.attack_alpha) * clean_loss)
                        outputs = clean_output  # For accuracy calculation
                    else:
                        # Standard training
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                    # Add proximal term if needed
                    if proximal_mu is not None:
                        proximal_term = self.model_dist(global_params_tensor, gradient_calc=True)
                        loss += (proximal_mu / 2) * proximal_term

                # Backward pass with gradient masking
                scaler.scale(loss).backward()
                
                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()
                
                # Project poisoned model parameters
                if self.atk_config.poisoned_is_projection and \
                    ( (batch_idx + 1) % self.atk_config.poisoned_projection_frequency == 0 or 
                     (batch_idx == len(self.train_loader) - 1) ):
                    # log(INFO, f"Client [{self.client_id}]: Projecting poisoned model parameters")
                    self._projection(global_params_tensor)

                running_loss += loss.item() * len(labels)
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_total += len(images)

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

        # Final evaluation
        self.train_backdoor_loss, self.train_backdoor_acc = self.poison.poison_test(
            self.model, 
            self.train_loader, 
            normalization=normalization
        )
        self.train_loss = epoch_loss
        self.train_accuracy = epoch_accuracy
        self.training_time = time.time() - start_time

        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Final Train Loss: {self.train_loss:.4f} | "
            f"Train Accuracy: {self.train_accuracy:.4f} | "
            f"Backdoor Loss: {self.train_backdoor_loss:.4f} | "
            f"Backdoor Accuracy: {self.train_backdoor_acc:.4f}")

        # Prepare return values
        if self.atk_config["scale_weights"]:
            state_dict = self.get_model_replacement_parameters(
                scale_factor=self.atk_config["scale_factor"],
                global_params=train_package["global_model_params"]
            )
        else:
            state_dict = self.get_model_parameters()

        training_metrics = {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "train_backdoor_loss": self.train_backdoor_loss,
            "train_backdoor_acc": self.train_backdoor_acc,
        }

        return len(self.train_dataset), state_dict, training_metrics

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model maliciously.
        """ 
        metrics = super().evaluate(test_package) # Dict of val_clean_loss, val_clean_acc
        if self.atk_config.backdoor_eval:
            backdoor_loss, backdoor_accuracy = self.poison_module.poison_test(self.model, self.val_loader, normalization=self.normalization)
            metrics['val_backdoor_loss'] = backdoor_loss
            metrics['val_backdoor_acc'] = backdoor_accuracy
        return len(self.val_dataset), metrics

    @torch.no_grad()
    def get_model_replacement_parameters(self, scale_factor: float, global_params: Dict[str, torch.Tensor]):
        """
        Model replacement update: Equation (3) in https://arxiv.org/pdf/1807.00459
        """
        model_params = {}
        for name, param in self.model.state_dict().items():
            global_param = global_params[name].to(self.device)
            local_param = param.to(self.device)
            model_params[name] = (global_param + scale_factor * (local_param - global_param)).cpu()
    
        return model_params

    def model_dist(self, global_params_tensor: torch.Tensor, gradient_calc=False):
        """Calculate the L2 distance between client model parameters and global parameters"""
        client_params_tensor = torch.cat([param.view(-1) for param in self.model.parameters()]).to(self.device)
        global_params_tensor = global_params_tensor.to(self.device)
        if gradient_calc:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2).item()
        else:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2)
        
    @torch.no_grad()
    def _projection(self, global_params_tensor: torch.Tensor):
        """Project model parameters to be within epsilon L2 ball of global parameters"""
        
        # Calculate L2 distance from global parameters
        client_params_tensor = torch.cat([param.view(-1) for param in self.model.parameters()]).to(self.device)
        model_dist_norm = torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2).item()
        
        # Project if distance exceeds epsilon
        if model_dist_norm > self.atk_config["poisoned_projection_eps"]:
            norm_scale = self.atk_config["poisoned_projection_eps"] / model_dist_norm
            projected_params = global_params_tensor + norm_scale * (client_params_tensor - global_params_tensor)
            
            # Update model parameters
            vector_to_parameters(projected_params, self.model.parameters())

class PoisonedDataset(Dataset):
    # Offline poisoning
    def __init__(self, dataset, poison_module, poison_ratio):
        self.dataset = dataset
        self.poison_module = poison_module
        
        indices = self.dataset.indices
        poison_indices = random.sample(indices, int(len(indices) * poison_ratio))
        self.poison_indices = poison_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index in self.poison_indices:
            return self.poison_module.poison_inputs(self.dataset[index])
        else:
            return self.dataset[index]

def train_poison_wrapper(poison_module):      
    def poison_collate_fn(batch):
        if not batch:  # Handle empty batch case
            return batch
        return poison_module.poison_batch(batch, mode="train")
    
    return poison_collate_fn
