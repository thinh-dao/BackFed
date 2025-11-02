import torch.nn as nn
import torch
import os
import time
import shutil

from omegaconf import DictConfig
from .base import Poison
from backfed.models import UNet, MNISTAutoencoder
from backfed.utils import log
from logging import INFO, WARNING

DEFAULT_PARAMS = {
    "atk_eps": 0.3,
    "atk_test_eps": 0.05,  # Target epsilon after decay
    "eps_decay_rate": 0.01,  # Decay rate per round
    "atk_lr": 0.01,
    "outter_epochs": 200,
    "save_atk_model_at_last": True,
}

class IBA(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1, **kwargs):
        super().__init__(params, client_id)
        
        # Merge default parameters with provided kwargs
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)

        self.sync_poison = True  # Sync poison resources across clients

        # Initialize local model
        if "NIST" in self.params.dataset.upper():  
            log(INFO, "Using MNISTAutoencoder for IBA")
            self.atk_model = MNISTAutoencoder().to("cuda")
            self.atk_model_name = "mnist_autoencoder"
        else:
            self.atk_model = UNet(3).to("cuda")    
            self.atk_model_name = "unet"

        if self.save_atk_model_at_last:
            self.atk_model_path = os.path.join("backfed/poisons/saved", "iba")
            os.makedirs(self.atk_model_path, exist_ok=True)
        
        # Epsilon decay tracking
        self.cur_eps = self.atk_eps  # Current epsilon
        self.decay_start_round = None  # Track when decay starts
    
    def exponential_decay(self, init_val, decay_rate, t):
        """Exponential decay: init_val * (1 - decay_rate)^t"""
        return init_val * (1.0 - decay_rate) ** t
    
    def update_epsilon(self, server_round):
        """Update current epsilon with decay"""
        if self.decay_start_round is None:
            self.decay_start_round = server_round
        
        t = server_round - self.decay_start_round
        decayed_eps = self.exponential_decay(self.atk_eps, self.eps_decay_rate, t)
        self.cur_eps = max(self.atk_test_eps, decayed_eps)
        
        log(INFO, f"Round {server_round}: cur_eps={self.cur_eps:.4f}")

    @torch.no_grad()
    def poison_inputs(self, inputs):
        noise = self.atk_model(inputs) * self.cur_eps
        return torch.clamp(inputs + noise, min=0, max=1)
    
    def poison_update(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """Update the trigger generator model"""
        # Update epsilon with decay
        self.update_epsilon(server_round)
        self.train_atk_model(client_id=client_id, 
                             server_round=server_round, 
                             model=initial_model, 
                             dataloader=dataloader, 
                             normalization=normalization)

    def train_atk_model(self, client_id, server_round, model, dataloader, normalization=None):
        log(INFO, f"Client [{client_id}]: Train IBA trigger generator in round {server_round}, atk_eps: {self.cur_eps}, target_class: {self.params.target_class}.")
        start_time = time.time()

        if len(dataloader) == 0:
            log(WARNING, f"Client [{client_id}]: Empty dataloader, returning current trigger")
            return self.trigger_image.detach()

        loss_fn = nn.CrossEntropyLoss()
        # training trigger
        model.eval()  # classifier model
        self.freeze_model(model)
        
        self.atk_model.train()  # trigger model
        num_attack_sample = -1  # poison all samples

        local_asr, threshold_asr = 0.0, 0.85  # Stop training if local ASR exceeds threshold
        atk_optimizer = torch.optim.Adam(self.atk_model.parameters(), lr=self.atk_lr)
        
        for atk_train_epoch in range(self.outter_epochs):
            if local_asr >= threshold_asr:
                log(INFO, f"Client [{client_id}]: Early stopping - threshold_asr reached ({local_asr:.4f} >= {threshold_asr})")
                break

            backdoor_preds, backdoor_loss, total_sample = 0, 0, 0
            
            for _, batch in enumerate(dataloader):
                inputs, labels = batch[0].to("cuda"), batch[1].to("cuda")
                
                # Zero gradients for the optimizer
                atk_optimizer.zero_grad()
                
                # Generate poisoned inputs using the attack model
                noise = self.atk_model(inputs) * self.cur_eps
                poisoned_inputs = torch.clamp(inputs + noise, min=0, max=1)
                poisoned_labels = self.poison_labels(labels)
                
                if normalization:
                    poisoned_inputs = normalization(poisoned_inputs)

                if num_attack_sample != -1:
                    poisoned_inputs = poisoned_inputs[:num_attack_sample]
                    poisoned_labels = poisoned_labels[:num_attack_sample]
                
                # Forward pass through the classifier model
                poisoned_outputs = model(poisoned_inputs)
                loss_p = loss_fn(poisoned_outputs, poisoned_labels)
                backdoor_loss += loss_p.item()
                
                # Backward pass
                loss_p.backward()
                atk_optimizer.step()

                backdoor_preds += (torch.max(poisoned_outputs.data, 1)[1] == poisoned_labels).sum().item()
                total_sample += len(poisoned_labels)

            local_asr = backdoor_preds / total_sample
            backdoor_loss = backdoor_loss / len(dataloader)
            if atk_train_epoch % 10 == 0:
                log(INFO, f"Epoch {atk_train_epoch} updated atk_model - local_asr: {local_asr*100:.2f}% | threshold_asr: {threshold_asr*100:.2f}% | backdoor_loss: {backdoor_loss}")
        
        self.unfreeze_model(model)
        end_time = time.time()
        log(INFO, f"Client [{client_id}]: Trigger generator training time: {end_time - start_time:.2f}s")

    def save_atk_model(self, name, server_round=None, path=None):
        """
        Save the attacker model for the poisoning round and keep track of the latest version.
        """
        if path is None:
            path = self.atk_model_path 
        
        if server_round is None:
            log(INFO, f"Saving Attacker Model")
            server_round = "latest"
        else: 
            log(INFO, f"Saving Attacker Model for round {server_round}")

        save_path = os.path.join(path, f"{name}_{server_round}.pt")
        torch.save(self.atk_model.state_dict(), save_path)
    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
    def get_shared_resources(self) -> dict:
        """
        Get the resources to be shared across clients in parallel mode.
        Returns:
            resources (dict): The resources to be shared
        """
        return {
            "atk_model_state_dict": {k: v.cpu() for k, v in self.atk_model.state_dict().items()},
            "decay_start_round": self.decay_start_round
        }
    
    def update_shared_resources(self, resources: dict):
        """
        Update the resources shared across clients in parallel mode.
        Args:
            resources (dict): The resources to be updated
        """
        self.atk_model.load_state_dict(resources["atk_model_state_dict"])
        self.decay_start_round = resources["decay_start_round"]

    def poison_finish(self):
        if self.save_atk_model_at_last:
            self.save_atk_model(name=self.atk_model_name)
