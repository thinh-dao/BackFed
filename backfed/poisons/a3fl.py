import os
import time
import torch
import torchvision
import copy
import shutil

from backfed.utils import log
from backfed.poisons.pattern import Pattern
from torch.nn import CrossEntropyLoss
from logging import INFO,WARNING

DEFAULT_PARAMS = {
    "trigger_outter_epochs": 100,
    "trigger_lr": 0.01,
    "dm_adv_epochs": 5,
    "dm_adv_K": 100,
    "dm_adv_model_count": 1,
    "noise_loss_lambda": 0.01,
    "save_trigger_at_last": True,
}

class A3FL(Pattern):
    def __init__(self, params, client_id: int = -1, **kwargs):
        super().__init__(params, client_id) # Sync poison resources across clients in parallel mode
        
        # Merge default parameters with provided kwargs
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)
        
        self.sync_poison = True  # Sync poison resources across clients

        self.trigger_image *= 0.5  # Follow the original implementation
        self.adversarial_loss_fn = CrossEntropyLoss()  # Default loss function for adversarial training

        if self.save_trigger_at_last:
            self.trigger_name = "a3fl_trigger" # Save name for the trigger image in trigger_path
            self.trigger_path = os.path.join("backfed/poisons/saved", "a3fl")
            os.makedirs(self.trigger_path, exist_ok=True)

    def poison_update(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """Update the adversarial trigger"""
        self.search_trigger(client_id=client_id, 
                            server_round=server_round, 
                            model=initial_model, 
                            dataloader=dataloader, 
                            normalization=normalization)

    def get_adv_model(self, model, dataloader, normalization=None):
        """
        Get the adversarially-trained model by training the model on poisoned inputs and ground-truth labels.
        """
        adv_model = copy.deepcopy(model)
        self.unfreeze_model(adv_model)
        adv_model.train()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)

        for _ in range(self.dm_adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()

                inputs = self.poison_inputs(inputs)
                if normalization:
                    inputs = normalization(inputs)

                outputs = adv_model(inputs)
                loss = self.adversarial_loss_fn(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].data.reshape(-1),\
                                    dict(model.named_parameters())[name].data.reshape(-1))
        
        # Manually zero the gradients of model parameters
        self.freeze_model(adv_model)
        adv_model.eval()
        return adv_model, sim_sum/sim_count
    
    def search_trigger(self, client_id, server_round, model, dataloader, normalization=None):
        """
        Fine-tune the trigger using gradient descent on poisoned samples.
        Args:
            client_id: The ID of the client
            server_round: The current server round
            model: The model to be used for trigger optimization
            dataloader: The dataloader providing training data
            normalization: Optional normalization function to be applied to inputs
        Returns:
            Optimized trigger tensor
        """
        log(INFO, f"Client [{client_id}]: Search trigger at server round {server_round}")
        start_time = time.time()
        
        # Validate dataloader
        if len(dataloader) == 0:
            log(WARNING, f"Client [{client_id}]: Empty dataloader, returning current trigger")
            return self.trigger_image.detach()
        
        self.freeze_model(model)
        model.eval()

        adv_models = []
        adv_weights = []        

        self.trigger_image.requires_grad = True # Enable gradients for trigger optimization
        ce_loss_fn = CrossEntropyLoss()

        local_asr, threshold_asr = 0.0, 0.85
        
        for trigger_train_epoch in range(self.trigger_outter_epochs):
            if local_asr >= threshold_asr:
                log(INFO, f"Client [{client_id}]: Early stopping - threshold_asr reached ({local_asr:.4f} >= {threshold_asr})")
                break

            backdoor_preds, total_loss, total_sample = 0, 0.0, 0
    
            # Periodically update adversarial models
            if trigger_train_epoch % self.dm_adv_K == 0 and trigger_train_epoch != 0:
                adv_models.clear()
                adv_weights.clear()
                
                # Create new adversarial models
                for _ in range(self.dm_adv_model_count):
                    adv_model, adv_weight = self.get_adv_model(model, dataloader, normalization=normalization) 
                    adv_models.append(adv_model)
                    adv_weights.append(adv_weight)

            for batch in dataloader:
                poison_inputs, poison_labels = super().poison_batch(batch, mode="test")
                
                if normalization:
                    poison_inputs = normalization(poison_inputs)

                # Zero gradients before backward pass
                if self.trigger_image.grad is not None:
                    self.trigger_image.grad.zero_()

                outputs = model(poison_inputs) 
                backdoor_loss = ce_loss_fn(outputs, poison_labels)

                adaptation_loss = 0
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_weight = adv_weights[am_idx]
                        adaptation_loss += adv_weight * ce_loss_fn(adv_model(poison_inputs), poison_labels)

                if len(adv_models) > 0:
                    loss = backdoor_loss + self.noise_loss_lambda/self.dm_adv_model_count * adaptation_loss
                else:
                    loss = backdoor_loss

                loss.backward()

                self.trigger_image.data -= self.trigger_lr * self.trigger_image.grad.sign()
                self.trigger_image.data.clamp_(0, 1)

                # Track metrics
                backdoor_preds += (torch.max(outputs.data, 1)[1] == poison_labels).sum().item()
                total_loss += backdoor_loss.item()
                total_sample += len(poison_labels)

            # Calculate epoch metrics
            local_asr = backdoor_preds / total_sample
            avg_loss = total_loss / len(dataloader)

            if trigger_train_epoch % 10 == 0:
                log(INFO, f"Epoch {trigger_train_epoch} - updated trigger: local_asr: {local_asr*100:.2f}% | threshold_asr: {threshold_asr*100:.2f}% | backdoor_loss: {avg_loss:.4f}")

        # Cleanup
        adv_models.clear()
        adv_weights.clear()
        
        self.unfreeze_model(model)
        self.trigger_image.requires_grad = False

        end_time = time.time()
        log(INFO, f"Client [{client_id}]: Trigger search completed in {end_time - start_time:.2f}s (final ASR: {local_asr:.4f})")
        return self.trigger_image.detach()

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
        }
    
    def update_shared_resources(self, resources: dict):
        """
        Update the resources shared across clients in parallel mode.
        Args:
            resources (dict): The resources to be updated
        """
        self.trigger_image = resources["trigger_image"].to(self.device)

    def poison_finish(self):
        if self.save_trigger_at_last:
            self.save_trigger(name=self.trigger_name)
