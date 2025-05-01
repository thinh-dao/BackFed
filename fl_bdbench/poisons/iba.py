import torch.nn as nn
import torch
import os
import time
import shutil

from omegaconf import DictConfig
from .base import Poison
from fl_bdbench.models import UNet, MNISTAutoencoder
from logging import INFO
from flwr.common.logger import log

DEFAULT_PARAMS = {
    "atk_eps": 0.06,
    "atk_lr": 0.01,
    "outter_epochs": 100,
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

        # Initialize local model
        if "NIST" in self.params.dataset.upper():  
            log(INFO, "Using MNISTAutoencoder for IBA")
            self.atk_model = MNISTAutoencoder().to("cuda")
            self.atk_model_name = "mnist_autoencoder"
        else:
            self.atk_model = UNet(3).to("cuda")    
            self.atk_model_name = "unet"

        self.atk_model_path = os.path.join("fl_bdbench/poisons/saved", "iba")
        os.makedirs(self.atk_model_path, exist_ok=True)

    @torch.no_grad()
    def poison_inputs(self, inputs):
        noise = self.atk_model(inputs) * self.atk_eps
        return torch.clamp(inputs + noise, min=0, max=1)
    
    def poison_warmup(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """Update the trigger generator model"""
        self.train_atk_model(client_id=client_id, 
                             server_round=server_round, 
                             model=initial_model, 
                             dataloader=dataloader, 
                             normalization=normalization)

    def train_atk_model(self, client_id, server_round, model, dataloader, normalization=None):
        log(INFO, f"Client [{client_id}]: Train IBA trigger generator in round {server_round}, atk_eps: {self.atk_eps}, target_class: {self.params.target_class}.")

        start_time = time.time()

        loss_fn = nn.CrossEntropyLoss()
        # training trigger
        model.eval()  # classifier model
        self.freeze_model(model)
        
        self.atk_model.train()  # trigger model
        num_attack_sample = -1  # poison all samples

        local_asr, threshold_asr = 0.0, 0.8
        atk_optimizer = torch.optim.Adam(self.atk_model.parameters(), lr=self.atk_lr)
        
        for atk_train_epoch in range(self.outter_epochs):
            if local_asr >= threshold_asr:
                break

            backdoor_preds, backdoor_loss, total_sample = 0, 0, 0
            
            for _, batch in enumerate(dataloader):
                inputs, labels = batch[0].to("cuda"), batch[1].to("cuda")
                
                # Zero gradients for the optimizer
                atk_optimizer.zero_grad()
                
                # Generate poisoned inputs using the attack model
                noise = self.atk_model(inputs) * self.atk_eps
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
                log(INFO, f"Epoch {atk_train_epoch}: local_asr: {local_asr} | threshold_asr: {threshold_asr} | backdoor_loss: {backdoor_loss}")
        
        self.unfreeze_model(model)
        end_time = time.time()
        log(INFO, f"Client [{client_id}]: Trigger generator training time: {end_time - start_time:.2f}s")


    def save_atk_model(self, name, server_round, path=None):
        """
        Save the attacker model for the poisoning round and keep track of the latest version.
        """
        log(INFO, f"Saving Attacker Model for round {server_round}")
        if path is None:
            path = self.atk_model_path 
        
        save_path = os.path.join(path, f"{name}_latest.pt")
        torch.save(self.atk_model.state_dict(), save_path)
        save_path = os.path.join(path, f"{name}_{server_round}.pt")
        torch.save(self.atk_model.state_dict(), save_path)
    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def poison_finish(self):
        if self.save_atk_model_at_last:
            save_path = os.path.join(os.getcwd(), "fl_bdbench/attack/saved/iba")
            os.makedirs(save_path, exist_ok=True)
            latest_atk_model = os.path.join(self.atk_model_path, f"{self.atk_model_name}_latest.pt")
            shutil.copy(latest_atk_model, os.path.join(save_path, f"IBA_{self.atk_model_name}.pt"))
            
        # Delete the attacker model of the last run
        if os.path.exists(self.atk_model_path):
            shutil.rmtree(self.atk_model_path)
