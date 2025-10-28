"""
Chameleon client implementation for FL.
"""

import torch
import torch.nn as nn

from backfed.clients.base_malicious_client import MaliciousClient
from backfed.models import SupConModel
from backfed.utils import log 
from logging import INFO, WARNING

DEFAULT_PARAMS = {
    "poison_supcon_retrain_no_times": 10,
    "poison_supcon_lr": 0.005,
    "poison_supcon_momentum": 0.9,
    "poison_supcon_weight_decay": 0.0005,
    "poison_supcon_milestones": [3, 5, 7, 9],
    "poison_supcon_lr_gamma": 0.1,
    "fac_scale_weight": 6,
}

class ChameleonClient(MaliciousClient):
    """
    Chameleon client implementation for FL.
    """

    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        atk_config,
        poison_module,
        context_actor,
        client_type: str = "chameleon_malicious",
        verbose: bool = True,
        **kwargs
    ):
        # Merge default parameters with provided params
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
        # Initialize the client. After this, additional kwargs are updated to atk_config
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
            **params_to_update
        )

    def train_contrastive_model(self, server_round, global_params_tensor=None, normalization=None, proximal_mu=None):
        """
        Train the model maliciously for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        """
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - Training contrastive model")
        
        # Initialize contrastive model
        self.contrastive_model = SupConModel(self.model)

        # Setup training protocol
        self.supcon_loss = SupConLoss().cuda()
        self.supcon_optimizer = torch.optim.SGD(self.contrastive_model.parameters(), 
                                                lr=self.atk_config["poison_supcon_lr"],
                                                momentum=self.atk_config["poison_supcon_momentum"], 
                                                weight_decay=self.atk_config["poison_supcon_weight_decay"]
                                            )   
        self.supcon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.supcon_optimizer,
                                                milestones=self.atk_config['poison_supcon_milestones'],
                                                gamma=self.atk_config['poison_supcon_lr_gamma']
                                            )

        # Training loop
        for internal_round in range(self.atk_config["poison_supcon_retrain_no_times"]):
            for batch_idx, batch in enumerate(self.train_loader):
                self.supcon_optimizer.zero_grad()

                data, targets = self.poison_module.poison_batch(batch, mode="train")
                if normalization:
                    data = normalization(data)

                output = self.contrastive_model(data)
                contrastive_loss = self.supcon_loss(output, targets,
                                                    scale_weight=self.atk_config["fac_scale_weight"],
                                                    fac_label=self.atk_config["target_class"])
                
                # Add proximal term if needed
                if proximal_mu is not None and global_params_tensor is not None:
                    distance_loss = super().model_dist(client_model=self.contrastive_model, global_params_tensor=global_params_tensor, gradient_calc=True)
                    loss = contrastive_loss + (proximal_mu/2) * distance_loss
                else:
                    loss = contrastive_loss

                # Check for NaN in total loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    log(WARNING, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                        f"NaN/Inf detected in total loss at epoch {internal_round}, batch {batch_idx}. Skipping batch.")
                    continue
                
                loss.backward()
                self.supcon_optimizer.step()
                
                # Project poisoned model parameters
                poison_projection = self.atk_config["poisoned_is_projection"] and (
                    (batch_idx + 1) % self.atk_config["poisoned_projection_frequency"] == 0 or 
                    (batch_idx == len(self.train_loader) - 1) 
                )
                if poison_projection:
                    self._projection(global_params_tensor)

            self.supcon_scheduler.step()
            
            if self.verbose and internal_round % (self.atk_config["poison_supcon_retrain_no_times"] // 5) == 0:
                self.contrastive_model.transfer_params(target_model=self.model)
                backdoor_total_samples, backdoor_loss, backdoor_accuracy = self.poison_module.poison_test(self.model, self.train_loader)
                backdoor_correct_preds = round(backdoor_accuracy * backdoor_total_samples)
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                    f"Epoch {internal_round} | Contrastive loss: {contrastive_loss.item()} | "
                    f"Backdoor loss: {backdoor_loss} | "
                    f"Backdoor accuracy: {backdoor_accuracy} ({backdoor_correct_preds}/{backdoor_total_samples})"
                )
    
    def train(self, train_package):
        """
        Train the model maliciously.
        """
        ######### Phase 1 - Train the contrastive model #########
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "normalization", "server_round", "global_state_dict"
        ])

        normalization = train_package["normalization"]
        server_round = train_package["server_round"]
        selected_malicious_clients = train_package["selected_malicious_clients"]
        global_state_dict = train_package["global_state_dict"]

        # Verify client is selected for poisoning
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"
        
        # Setup training protocol
        proximal_mu = train_package.get('proximal_mu', None) if self.atk_config.follow_protocol else None
        if self.atk_config.poisoned_is_projection or proximal_mu is not None:
            global_params_tensor = torch.cat([param.view(-1).detach().clone().requires_grad_(False) for name, param in global_state_dict.items()
                                  if "weight" in name or "bias" in name]).to(self.device)
        else:
            global_params_tensor = None
        
        # Update local model
        self.model.load_state_dict(global_state_dict)

        # Initialize poison attack
        if self.poison_module.sync_poison: # If poison module requires synchronization
            self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)

        # Train contrastive model
        self.model.train()  
        self.train_contrastive_model(
            server_round=server_round, 
            global_params_tensor=global_params_tensor, 
            normalization=normalization, 
            proximal_mu=proximal_mu
        )

        # Transfer the trained weights of encoder to the local model and freeze the encoder
        self.contrastive_model.transfer_params(target_model=self.model)

        last_layer_names = ["linear", "fc", "classifier"]
        for params in self.model.named_parameters():
            if any(name in params[0] for name in last_layer_names):
                params[1].requires_grad = True
            else:
                params[1].requires_grad = False


        ######### Phase 2 - Train the linear layer #########

        # Setup poisoned dataloader if poison_mode is offline
        if self.atk_config.poison_mode == "offline":
            self._set_poisoned_dataloader()

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
                if self.atk_config.poison_mode == "multi_task": # IBA style
                    # Handle multi-task poisoning
                    clean_images = images.detach().clone()
                    clean_labels = labels.detach().clone()
                    poisoned_images = self.poison_module.poison_inputs(images)
                    poisoned_labels = self.poison_module.poison_labels(labels)

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
                    outputs = poisoned_output  # For accuracy calculation, focus on poisoned output

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
                    proximal_term = self.model_dist(global_params_tensor=global_params_tensor, gradient_calc=True)
                    loss += (proximal_mu / 2) * proximal_term

                # Backward pass
                loss.backward()

                # Optimizer step
                self.optimizer.step()

                # Project poisoned model parameters
                poison_projection = self.atk_config["poisoned_is_projection"] and (
                    (batch_idx + 1) % self.atk_config["poisoned_projection_frequency"] == 0 or 
                    (batch_idx == len(self.train_loader) - 1) 
                )
                if poison_projection:
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

        train_loss = epoch_loss
        train_acc = epoch_accuracy

        # Unfreeze the model
        for params in self.model.parameters():
            params.requires_grad = True

        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
            f"Train Backdoor Loss: {train_loss:.4f} | "
            f"Train Backdoor Accuracy: {train_acc:.4f} | "
        )

        training_metrics = {
            "train_backdoor_loss": train_loss,
            "train_backdoor_acc": train_acc,
        }

        model_updates = self.weight_diff_dict(client_state_dict=self.model.state_dict(), 
                                              global_state_dict=train_package["global_state_dict"]
                                            )
        
        if self.atk_config["scale_poison"]:
            self.model_replacement_inplace(
                scale_factor=self.atk_config["scale_factor"],
                model_updates=model_updates 
            )

        return len(self.train_dataset), model_updates, training_metrics

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: 
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, scale_weight=1, fac_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            
            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)
            
            for ind, label in enumerate(labels.view(-1)):
                if label == fac_label:
                    mask_scale[ind, :] = mask[ind, :] * scale_weight

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature 
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss
