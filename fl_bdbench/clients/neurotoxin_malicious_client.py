"""
Chameleon client implementation for FL.
"""

import torch
import torch.nn as nn
import time
import copy
import numpy as np

from torch.utils.data import DataLoader
from fl_bdbench.clients.base_malicious_client import MaliciousClient, PoisonedDataset, train_poison_wrapper
from fl_bdbench.utils import log 
from fl_bdbench.utils import model_dist_layer
from fl_bdbench.poisons import Poison
from logging import INFO

DEFAULT_PARAMS = {
    "gradient_mask_ratio": 0.99, # Mask ratio - Project gradient into bottom 99% of the gradient space
    "aggregate_all_layers": True # Mask the aggregated model's gradients from a concatenated vector or mask layer by layer
}

class NeurotoxinClient(MaliciousClient):
    """
    Neurotoxin client implementation for FL.
    """

    def __init__(self, client_id, dataset, dataset_indices, model, client_config, atk_config, poison_module, context_actor, **kwargs):
        """
        Initialize the Neurotoxin client.
        """
        # Merge default parameters with provided params
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            dataset_indices=dataset_indices,
            model=model,
            client_config=client_config,
            atk_config=atk_config,
            poison_module=poison_module,
            context_actor=context_actor,
            client_type="neurotoxin",
            **params_to_update
        )
        
    def train(self, train_package):
        """
        Train the model maliciously for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        """
        assert "poison_module" in train_package, "No poison module provided. Using benign training."

        start_time = time.time()
        self.poison: Poison = train_package["poison_module"]
        self.model.load_state_dict(train_package["global_model_params"])
        selected_malicious_clients = train_package["selected_malicious_clients"]
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)
        
        # Poison warmup
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"
        self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)

        # Prepare poisoned dataloader
        super()._set_poisoned_dataloader()

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

        # The key difference between NeurotoxinClient and MaliciousClient
        mask_grad_list = self._compute_grad_mask(self.model, self.train_loader, ratio=self.atk_config.gradient_mask_ratio)
            
        if self.atk_config.poison_until_convergence:
            num_epochs = 100 # large number of epochs to train until convergence
            log(INFO, f"Client [{self.client_id}]: Training until convergence of backdoor loss")
        else:
            num_epochs = self.atk_config.poison_epochs

        # Training loop
        self.model.train()
        for internal_epoch in range(num_epochs):
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
                    poisoned_images, poisoned_labels = self.poison.poison_inputs(clean_images), self.poison.poison_labels(clean_labels)

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
                self._apply_grad_mask(mask_grad_list)
                scaler.step(self.optimizer)
                scaler.update()
                
                if self.atk_config.poisoned_is_projection and \
                    ( (batch_idx + 1) % self.atk_config.poisoned_projection_frequency == 0 or \
                    batch_idx == len(self.train_loader) - 1 ):
                    # log(INFO, f"Client [{self.client_id}]: Projecting poisoned model parameters")
                    self._projection(global_params_tensor)

                running_loss += loss.item() 

            training_loss = running_loss / len(self.train_loader)
            
            if self.verbose:
                backdoor_loss, backdoor_accuracy = self.poison.poison_test(self.model, self.train_loader, normalization=self.normalization)
                log(INFO, f"Malicious Client [{self.client_id}]: Epoch {internal_epoch} - Train Loss: {training_loss} - Backdoor Loss: {backdoor_loss} - Backdoor Accuracy: {backdoor_accuracy}")

            if self.atk_config["poison_until_convergence"] and backdoor_loss < self.atk_config["poison_convergence_threshold"]:
                break

            if self.atk_config["step_scheduler"]:
                scheduler.step()

        self.train_backdoor_loss, self.train_backdoor_acc = self.poison.poison_test(self.model, self.train_loader, normalization=self.normalization)
        self.train_loss = training_loss
        self.train_accuracy = self.train_backdoor_acc
        self.training_time = time.time() - start_time

        log(INFO, f"Malicious Client {self.client_id}: Train Loss: {training_loss} - Backdoor Loss: {self.train_backdoor_loss} - Backdoor Accuracy: {self.train_backdoor_acc}")

        if self.atk_config["scale_weights"]:
            submitted_parameters = self.get_model_replacement_parameters(scale_factor=self.atk_config["scale_factor"], global_params=train_package["global_model_params"])
        else:
            submitted_parameters = self.model.state_dict()

        if torch.cuda.is_available():
            self.memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        num_examples = len(self.train_dataset)
        metrics = {
            "client_id": self.client_id,
            "client_type": self.client_type,
            "train_backdoor_loss": self.train_backdoor_loss,
            "train_backdoor_acc": self.train_backdoor_acc,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage
        }

        return num_examples, submitted_parameters, metrics
    
    def _compute_grad_mask(self, model, clean_train_loader, criterion=torch.nn.CrossEntropyLoss(), ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        INTERNAL_ROUNDS = 5
        # log(INFO, f"Computing gradient mask with {INTERNAL_ROUNDS} rounds.")
        for _ in range(INTERNAL_ROUNDS):
            for inputs, labels in clean_train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        if self.atk_config.aggregate_all_layers:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    grad_list.append(param.grad.abs().view(-1))
                    grad_abs_sum_list.append(param.grad.abs().view(-1).sum().item())
                    k_layer += 1

            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            grad_sum = np.sum(grad_abs_sum_list)
            for _, param in model.named_parameters():
                if param.requires_grad:
                    gradients_length = len(param.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                    mask_grad_list.append(mask_flat.reshape(param.grad.size()).cuda())

                    count += gradients_length
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/grad_sum)
                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    grad_res.append(param.grad.view(-1))
                    l2_norm_l = torch.norm(param.grad.view(-1).clone().detach().cuda())/float(len(param.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            percentage_mask_list = []
            k_layer = 0
            for _, param in model.named_parameters():
                if param.requires_grad:
                    gradients = param.grad.abs().view(-1)
                    gradients_length = len(gradients)

                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(param.grad.size()).cuda())
                    
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)

                    k_layer += 1

        model.zero_grad()
        return mask_grad_list

