"""
Malicious client implementation for FL.
"""
import random
import time
import torch
import copy
import ray

from logging import INFO
from typing import List
from fl_bdbench.utils import log
from fl_bdbench.poisons import IBA, A3FL
from fl_bdbench.context_actor import ContextActor
from fl_bdbench.clients.base_benign_client import BenignClient
from fl_bdbench.utils import model_dist_layer
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from torch.nn.utils import vector_to_parameters
from omegaconf import DictConfig
from fl_bdbench.const import StateDict, Metrics
from typing import Tuple, Dict, Any

class MaliciousClient(BenignClient):
    """
    Malicious client implementation for FL.
    """ 

    def __init__(self, client_id, dataset, dataset_indices, model, client_config, atk_config, poison_module, context_actor, **kwargs):
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

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        """
        Train the model maliciously for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        """
        assert "poison_module" in train_package, "No poison module provided for malicious client"

        start_time = time.time()
        self.model.load_state_dict(train_package["global_model_params"])
        selected_malicious_clients = train_package["selected_malicious_clients"]
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
        
        # Poison warmup
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"
        self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)

        # Prepare poisoned dataloader
        self._set_poisoned_dataloader()

        # Set up training protocol
        if self.atk_config.follow_protocol and self.atk_config.get('proximal_mu', None) is not None:
            proximal_mu = self.atk_config['proximal_mu']
            self.atk_config['proximal_mu'] = proximal_mu
        else:
            proximal_mu = None

        scaler = torch.amp.GradScaler(device=self.device)

        if self.atk_config.poisoned_is_projection:
            global_params_tensor = [param.requires_grad(False) for name, param in train_package["global_model_params"].items() if "weight" in name or "bias" in name]

        if self.atk_config["step_scheduler"]:
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.atk_config["step_size"], gamma=0.1)
            
        if self.atk_config.poison_until_convergence:
            num_epochs = 100 # large number of epochs to train until convergence
            log(INFO, f"Client [{self.client_id}]: Training until convergence of backdoor loss")
        else:
            num_epochs = self.atk_config.poison_epochs

        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if len(labels) == 1: # Skip batch with only one sample
                    continue
                
                self.optimizer.zero_grad()
                
                images, labels = images.to(self.device), labels.to(self.device)
                if self.normalization:
                    images = self.normalization(images)

                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    if proximal_mu is not None:
                        proximal_term = model_dist_layer(self.model, global_params_tensor)
                        loss += (proximal_mu / 2) * proximal_term
                            
                if self.atk_config.poison_type == "combined-loss":
                    # Poison all the data and balance between clean loss and poisoned loss
                    clean_images, clean_labels = copy.deepcopy(images), copy.deepcopy(labels)
                    poisoned_images, poisoned_labels = self.poison_module.poison_inputs(clean_images), self.poison_module.poison_labels(clean_labels)

                    with torch.amp.autocast("cuda"):
                        clean_output = self.model(clean_images)
                        clean_loss = self.criterion(clean_output, clean_labels)

                        poisoned_output = self.model(poisoned_images)
                        poisoned_loss = self.criterion(poisoned_output, poisoned_labels)

                        loss = self.atk_config.attack_alpha * poisoned_loss + (1 - self.atk_config.attack_alpha) * clean_loss
                        if proximal_mu is not None:
                            proximal_term = model_dist_layer(self.model, global_params_tensor)
                            loss += (proximal_mu / 2) * proximal_term

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                if self.atk_config.poisoned_is_projection and \
                    ( (batch_idx + 1) % self.atk_config.poisoned_projection_frequency == 0 or \
                    batch_idx == len(self.train_loader) - 1 ):
                    # log(INFO, f"Client [{self.client_id}]: Projecting poisoned model parameters")
                    self._projection(global_params_tensor)

                running_loss += loss.item() 

            training_loss = running_loss / len(self.train_loader)

            if self.atk_config["poison_until_convergence"] and training_loss < self.atk_config["poison_convergence_threshold"]:
                break

            if self.atk_config["step_scheduler"]:
                scheduler.step()

        self.train_backdoor_loss, self.train_backdoor_acc = self.poison_module.poison_test(self.model, self.train_loader, normalization=self.normalization)
        self.train_loss = training_loss
        self.train_accuracy = self.train_backdoor_acc
        self.training_time = time.time() - start_time

        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) - Train Loss: {training_loss} - Backdoor Loss: {self.train_backdoor_loss} - Backdoor Accuracy: {self.train_backdoor_acc}")

        if self.atk_config["scale_weights"]:
            submitted_parameters = self.get_model_replacement_parameters(scale_factor=self.atk_config["scale_factor"], global_params=train_package["global_model_params"])
        else:
            submitted_parameters = self.get_model_parameters()

        num_examples = len(self.train_dataset)
        metrics = {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "train_backdoor_loss": self.train_backdoor_loss,
            "train_backdoor_acc": self.train_backdoor_acc,
        }

        return num_examples, submitted_parameters, metrics

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model maliciously.
        """ 
        num_examples, metrics = super().evaluate(test_package) # Dict of val_clean_loss, val_clean_acc
        if self.atk_config.backdoor_eval:
            backdoor_loss, backdoor_accuracy = self.poison_module.poison_test(self.model, self.val_loader, normalization=self.normalization)
            metrics['val_backdoor_loss'] = backdoor_loss
            metrics['val_backdoor_acc'] = backdoor_accuracy
        return num_examples, metrics

    def get_model_replacement_parameters(self, scale_factor: float, global_params: List[torch.Tensor]):
        """
        Model replacement update: Equation (3) in https://arxiv.org/pdf/1807.00459
        """
        model_params = {name: param.clone().cpu() for name, param in self.model.state_dict().items()}

        for name, param in model_params.items():
            model_params[name] = global_params[name] + scale_factor * (param - global_params[name])
    
        return model_params

    def _projection(self, global_params_tensor: torch.Tensor):
        # Do a l2 projection on the model parameters
        client_params_tensor = torch.cat([param.view(-1).to("cuda") for param in self.model.parameters()])
        with torch.no_grad():
            model_dist_norm = torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2).item()
            if model_dist_norm > self.atk_config["poisoned_projection_eps"]:
                norm_scale = self.atk_config["poisoned_projection_eps"] / model_dist_norm
                w_proj_vec = norm_scale * (client_params_tensor - global_params_tensor) + global_params_tensor

                # plug w_proj back into model
                vector_to_parameters(w_proj_vec, self.model.parameters())

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
