"""
Neurotoxin client implementation for FL.
"""

import torch
import time
import copy
import numpy as np
from functools import lru_cache
from torch.utils.data import DataLoader, Subset, RandomSampler

from fl_bdbench.clients.base_malicious_client import MaliciousClient
from fl_bdbench.utils import log
from logging import INFO, WARNING
from typing import List, Dict, Optional, Tuple, Any

DEFAULT_PARAMS = {
    "gradient_mask_ratio": 0.99, # Mask ratio - Project gradient into bottom 99% of the gradient space
    "aggregate_all_layers": True, # Mask the aggregated model's gradients from a concatenated vector or mask layer by layer
    "mask_compute_subset": 0.3,   # Use only a subset of data for mask computation (0.3 = 30%)
    "mask_compute_rounds": 2,     # Number of rounds for mask computation (reduced from 5)
    "skip_backdoor_eval": False,  # Skip backdoor evaluation during training
    "num_workers": 4,             # Number of workers for data loading
    "persistent_workers": True,   # Keep workers alive between iterations
    "prefetch_factor": 2,         # Number of batches to prefetch
}

class NeurotoxinClient(MaliciousClient):
    """
    Neurotoxin client implementation for FL.
    """

    def __init__(
        self,
        client_id,
        dataset,
        dataset_indices,
        model,
        client_config,
        atk_config,
        poison_module,
        context_actor,
        client_type: str = "neurotoxin_malicious",
        **kwargs
    ):
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
            client_type=client_type,
            **params_to_update
        )

        # Cache for gradient masks to avoid recomputation
        self._grad_mask_cache = {}

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
                - training_metrics: Dict containing training metrics
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
        normalization = train_package.get("normalization", None)

        # Verify client is selected for poisoning
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"

        # Initialize poison attack
        self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)

        # Compute gradient mask (key feature of Neurotoxin)
        mask_grad_list = self._compute_grad_mask(
            self.model,
            self.train_loader,
            ratio=self.atk_config.gradient_mask_ratio
        )

        # Setup poisoned dataloader if poison_mode is offline
        if self.atk_config.poison_mode == "offline":
            super().set_poisoned_dataloader()

        # Setup training protocol
        proximal_mu = self.atk_config.get('proximal_mu', None) if self.atk_config.follow_protocol else None

        # Initialize training tools
        scaler = torch.amp.GradScaler(device=self.device)

        if self.atk_config.poisoned_is_projection or proximal_mu is not None:
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

                # Zero gradients
                self.optimizer.zero_grad()

                # Prepare batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass and loss computation
                with torch.amp.autocast("cuda"):
                    if self.atk_config.poison_mode == "multi_task":
                        # Handle multi-task poisoning
                        clean_images = images.clone()
                        clean_labels = labels.clone()
                        poisoned_images = self.poison_module.poison_inputs(clean_images)
                        poisoned_labels = self.poison_module.poison_labels(clean_labels)

                        # Apply normalization if provided
                        if normalization:
                            clean_images = normalization(clean_images)
                            poisoned_images = normalization(poisoned_images)

                        # Compute losses for both clean and poisoned data in a single forward pass
                        clean_output = self.model(clean_images)
                        poisoned_output = self.model(poisoned_images)

                        clean_loss = self.criterion(clean_output, clean_labels)
                        poisoned_loss = self.criterion(poisoned_output, poisoned_labels)

                        # Combine losses according to attack alpha
                        loss = (self.atk_config.attack_alpha * poisoned_loss +
                               (1 - self.atk_config.attack_alpha) * clean_loss)

                    elif self.atk_config.poison_mode in ["online", "offline"]:
                        if self.atk_config.poison_mode == "online":
                            images, labels = self.poison_module.poison_batch(batch=(images, labels))

                        # Normalize images if needed
                        if normalization:
                            images = normalization(images)

                        # Forward pass and loss computation
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)

                    else:
                        raise ValueError(
                            f"Invalid poison_mode: {self.atk_config.poison_mode}. "
                            f"Expected one of: ['multi_task', 'online', 'offline']"
                        )

                    # Add proximal term if needed
                    if proximal_mu is not None:
                        proximal_term = self.model_dist(global_params_tensor, gradient_calc=True)
                        loss += (proximal_mu / 2) * proximal_term

                # Backward pass with gradient masking
                scaler.scale(loss).backward()
                self._apply_grad_mask(mask_grad_list)

                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()

                # Project poisoned model parameters
                if self.atk_config.poisoned_is_projection and \
                    ( (batch_idx + 1) % self.atk_config.poisoned_projection_frequency == 0 or (batch_idx == len(self.train_loader) - 1) ):
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
        self.train_backdoor_loss, self.train_backdoor_acc = self.poison_module.poison_test(
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

    def _compute_grad_mask(self, model, clean_train_loader, criterion=torch.nn.CrossEntropyLoss(), ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        # Accumulate gradients for each layer
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
                    l2_norm_l = torch.norm(param.grad.view(-1).detach().clone().cuda())/float(len(param.grad.view(-1)))
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

    ### Neurotoxin gradient masking function
    @torch.no_grad()
    def _apply_grad_mask(self, mask_grad_list: List[torch.Tensor]):
        """Optimized gradient mask application"""
        for (param, mask) in zip(
            (p for p in self.model.parameters() if p.requires_grad),
            mask_grad_list
        ):
            param.grad.mul_(mask)
