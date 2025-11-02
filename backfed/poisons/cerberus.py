"""
Trigger for "Poisoning with Cerberus: Stealthy and Colluded Backdoor Attack against Federated Learning".
Code referenced from: https://github.com/xtLyu/CerP/blob/main/cifar100_trigger.py
"""

import os
import time
import torch
import torchvision
import shutil
import numpy as np

from backfed.utils import log
from backfed.poisons.pattern import Pattern
from torch.nn import CrossEntropyLoss
from logging import INFO, WARNING

DEFAULT_PARAMS = {
    "trigger_lr": 0.1,
    "save_trigger_at_last": True,
}

class Cerberus(Pattern):
    def __init__(self, params, client_id: int = -1, **kwargs):
        super().__init__(params, client_id) # Sync poison resources across clients in parallel mode
        
        # Merge default parameters with provided kwargs
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)

        self.sync_poison = True  # Sync poison resources across clients
        self.initial_trigger = self.trigger_image.detach().clone() # We keep track of initial trigger
        self.adversarial_loss_fn = CrossEntropyLoss()  # Default loss function for adversarial training

        if self.save_trigger_at_last:
            self.trigger_name = "cerberus_trigger" # Save name for the trigger image in trigger_path
            self.trigger_path = os.path.join("backfed/poisons/saved", "cerberus")
            os.makedirs(self.trigger_path, exist_ok=True)

    def poison_update(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """Update the adversarial trigger"""
        self.search_trigger(client_id=client_id, 
                            server_round=server_round, 
                            model=initial_model, 
                            dataloader=dataloader, 
                            normalization=normalization)

    def search_trigger(self, client_id, server_round, model, dataloader, normalization=None):
        """
        Fine-tune the trigger using gradient descent on poisoned samples.
        
        Args:
            client_id: The ID of the client
            server_round: The current server round
            model: The model to attack
            dataloader: DataLoader for training data
            normalization: Optional normalization transform
            
        Returns:
            Optimized trigger tensor
        """
        log(INFO, f"Client [{client_id}]: Search trigger at server round {server_round}")
        start_time = time.time()

        if len(dataloader) == 0:
            log(WARNING, f"Client [{client_id}]: Empty dataloader, returning current trigger")
            return self.trigger_image.detach()
    
        self.freeze_model(model)
        model.eval()
        self.trigger_image.requires_grad = True # Enable gradients for trigger optimization
        
        backdoor_preds, backdoor_loss, total_sample = 0, 0.0, 0
        # Fine-tuning for one epoch
        for batch in dataloader:
            if self.trigger_image.grad is not None:
                self.trigger_image.grad.zero_()
        
            poison_inputs, poison_labels = super().poison_batch(batch, mode="test")
        
            if normalization:
                poison_inputs = normalization(poison_inputs)
            
            # Forward pass
            outputs = model(poison_inputs)
            loss = self.adversarial_loss_fn(outputs, poison_labels)
            
            # Backward pass
            loss.backward()
            
            # Update trigger with gradient descent (wrapped in no_grad for clarity)
            with torch.no_grad():
                self.trigger_image.data -= self.trigger_lr * self.trigger_image.grad
                
                # Project back to epsilon ball and clamp to [0,1]
                delta_trigger = self.trigger_image.data - self.initial_trigger
                self.trigger_image.data = self.initial_trigger + self.proj_lp(delta_trigger, xi=10, p=2)
                self.trigger_image.data.clamp_(0, 1)

            backdoor_preds += (torch.max(outputs.data, 1)[1] == poison_labels).sum().item()
            backdoor_loss += loss.item()
            total_sample += len(poison_labels)
        
        local_asr = backdoor_preds / total_sample if total_sample > 0 else 0.0
        total_loss = backdoor_loss / len(dataloader)
        log(INFO, f"Client [{client_id}] updated trigger - local_asr: {local_asr*100:.2f}% | backdoor_loss: {total_loss:.4f}")

        self.unfreeze_model(model)
        self.trigger_image.requires_grad = False  # Disable gradients after optimization

        end_time = time.time()
        log(INFO, f"Client [{client_id}]: Trigger search time: {end_time - start_time:.2f}s")
        return self.trigger_image.detach()
    
    def proj_lp(self, v, xi, p):
        """
        Project on the lp ball centered at 0 and of radius xi.
        
        Args:
            v: Vector to project
            xi: Radius of the lp ball (epsilon)
            p: Norm type (2 or np.inf)
            
        Returns:
            Projected vector
        """
        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / torch.norm(v))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return v

    def save_trigger(self, name, server_round=None, path=None):
        """
        Saving the trigger image in .pt and .png formats.
        """
        if path is None:
            path = self.trigger_path
        
        if server_round is None:
            log(INFO, f"Saving Trigger")
            server_round = "latest"
        else:
            log(INFO, f"Saving Trigger for round {server_round}")

        save_path = os.path.join(path, f"{name}_{server_round}.pt")
        torch.save(self.trigger_image, save_path)
        save_path_png = os.path.join(path, f"{name}_{server_round}.png")
        torchvision.utils.save_image(self.trigger_image * self.trigger_image_weight, save_path_png)

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
            "trigger_image": self.trigger_image.detach().clone().cpu(),
            "initial_trigger": self.initial_trigger.detach().clone().cpu(),
        }
    
    def update_shared_resources(self, resources: dict):
        """
        Update the resources shared across clients in parallel mode.
        Args:
            resources (dict): The resources to be updated
        """
        self.trigger_image = resources["trigger_image"].to(self.device)
        self.initial_trigger = resources["initial_trigger"].to(self.device)

    def poison_finish(self):
        if self.save_trigger_at_last:
            self.save_trigger(name=self.trigger_name)
