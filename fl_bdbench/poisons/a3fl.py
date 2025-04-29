import os
import time
import torch
import torchvision
import copy
import shutil

from fl_bdbench.utils import log
from fl_bdbench.poisons.pattern import Pattern
from torch.nn import CrossEntropyLoss
from logging import INFO

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
        super().__init__(params, client_id)
        
        # Merge default parameters with provided kwargs
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)
            
        self.trigger_image *= 0.5  # Follow the original implementation
        self.adversarial_loss_fn = CrossEntropyLoss()  # Default loss function for adversarial training
        self.trigger_name = "a3fl_trigger" # Save name for the trigger image in trigger_path
        self.trigger_path = os.path.join(self.params.output_dir, "a3fl")
        os.makedirs(self.trigger_path, exist_ok=True)

    def poison_warmup(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
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
        trigger_image = self.trigger_image.detach().clone()
        self.unfreeze_model(adv_model)
        adv_model.train()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.dm_adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs * (1-self.trigger_image_weight) + trigger_image * self.trigger_image_weight
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
        log(INFO, f"Client [{client_id}]: Search trigger at server round {server_round}")
        start_time = time.time()
        self.freeze_model(model)
        model.eval()

        adv_models = []
        adv_weights = []        
        alpha = self.trigger_lr

        self.trigger_image.requires_grad = True
        ce_loss_fn = CrossEntropyLoss()

        num_attack_sample = -1
        local_asr, threshold_asr = 0.0, 0.80
        
        for trigger_train_epoch in range(self.trigger_outter_epochs):
            if local_asr > threshold_asr:
                break

            backdoor_preds, backdoor_loss, total_sample = 0, 0, 0
    
            if trigger_train_epoch % self.dm_adv_K == 0 and trigger_train_epoch != 0:
                adv_models.clear()
                adv_weights.clear()
                for _ in range(self.dm_adv_model_count):
                    adv_model, adv_weight = self.get_adv_model(model, dataloader, normalization=normalization) 
                    adv_models.append(adv_model)
                    adv_weights.append(adv_weight)

            for batch in dataloader:
                poison_inputs, poison_labels = super().poison_batch(batch, mode="train")
                if normalization:
                    poison_inputs = normalization(poison_inputs)

                if num_attack_sample != -1:
                    poison_inputs = poison_inputs[:num_attack_sample]
                    poison_labels = poison_labels[:num_attack_sample]

                outputs = model(poison_inputs) 
                backdoor_loss = ce_loss_fn(outputs, poison_labels)

                adaptation_loss = 0
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_weight = adv_weights[am_idx]
                        adaptation_loss += adv_weight * ce_loss_fn(adv_model(poison_inputs), poison_labels)

                # Reset gradients before backward pass
                if self.trigger_image.grad is not None:
                    self.trigger_image.grad.zero_()

                loss = backdoor_loss + self.noise_loss_lambda/self.dm_adv_model_count * adaptation_loss
                loss.backward()
                self.trigger_image.data = self.trigger_image.data - alpha * self.trigger_image.grad.sign()
                self.trigger_image.data = torch.clamp(self.trigger_image.data, min=0, max=1)

                backdoor_preds += (torch.max(outputs.data, 1)[1] == poison_labels).sum().item()
                total_sample += len(poison_labels)

            local_asr = backdoor_preds / total_sample
            backdoor_loss = backdoor_loss / len(dataloader)

            if trigger_train_epoch % 10 == 0:
                log(INFO, f"Epoch {trigger_train_epoch}: local_asr: {local_asr} | threshold_asr: {threshold_asr} | backdoor_loss: {backdoor_loss}")

        self.unfreeze_model(model)
        end_time = time.time()
        log(INFO, f"Client [{client_id}]: Trigger search time: {end_time - start_time:.2f}s")

    def save_trigger(self, name, server_round, path=None):
        """
        Saving the trigger image in different .pt and .png formats.
        In .pt format: We save 2 versions, latest.pt and {server_round}.pt
        """
        if path is None:
            path = self.trigger_path

        save_path = os.path.join(path, f"{name}_{server_round}.png")
        torch.save(self.trigger_image, save_path)
        torchvision.utils.save_image(self.trigger_image * self.trigger_image_weight, save_path)
        save_path = os.path.join(path, f"{name}_{server_round}.pt")
        torch.save(self.trigger_image, save_path)
    
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def poison_finish(self):
        # Delete the trigger folder of the last run
        if self.save_trigger_at_last:
            save_path = os.path.join(os.getcwd(), "fl_bdbench/attack/saved/a3fl")
            os.makedirs(save_path, exist_ok=True)
            latest_trigger = os.path.join(self.trigger_path, f"{self.trigger_name}_latest")
            shutil.copy(latest_trigger + ".pt", os.path.join(save_path, f"{self.trigger_name}.pt"))
            shutil.copy(latest_trigger + ".png", os.path.join(save_path, f"{self.trigger_name}.png"))
            
        if os.path.exists(self.trigger_path):
            shutil.rmtree(self.trigger_path)
