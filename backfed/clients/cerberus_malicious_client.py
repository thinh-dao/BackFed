"""
Cerberus malicious client implementation for FL.

This client follows the reference CerP training procedure:
- Normal training (clean) to obtain a reference parameter set
- Poison training with an additional distance regularization to the normal model
- Optional _projection and weight scaling following configuration

Poison trigger is provided by backfed.poisons.cerberus.Cerberus and is fine-tuned via poison_update().
"""

import copy
import torch

from logging import INFO
from typing import Tuple, Dict, Any
from hydra.utils import instantiate
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.utils import log

DEFAULT_PARAMS = {
    # CerP-style losses
    "alpha_loss": 0.0001,  # weight for distance-to-clean-model loss
    "beta_loss": 0.0,   # weight for collusion cosine-similarity term (not used in this implementation)
}

class CerberusMaliciousClient(MaliciousClient):
    """
    Cerberus client implementation for FL.
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
        client_type: str = "cerberus_malicious",
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

    def _params_distance_norm(self, reference_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute L2 distance between current model parameters and reference parameters.

        Args:
            reference_params: Dict[name, tensor] (detached tensors from normal model)
        Returns:
            torch.Tensor: L2 distance (differentiable w.r.t. self.model parameters)
        """
        distance = torch.zeros(1, device=self.device)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            ref = reference_params.get(name, None)
            if ref is None:
                continue
            # ref is detached; subtraction stays differentiable w.r.t. param
            distance = distance + torch.norm(param - ref.to(self.device), p=2)
        return distance

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train the Cerberus malicious client.

        Args:
            train_package (dict): Contains training parameters including:
                - global_state_dict: Global model parameters
                - selected_malicious_clients: List of selected malicious clients
                - server_round: Current server round
                - normalization: Optional normalization function

        Returns:
            tuple: (num_examples, client_updates, training_metrics)
                - num_examples (int): number of examples in the training dataset
                - state_dict (ModelUpdate): updated model parameters
                - training_metrics (Dict[str, float]): training metrics
        """
        # Validate required keys
        self._check_required_keys(train_package, required_keys=[
            "global_state_dict", "selected_malicious_clients", "server_round"
        ])

        # Setup training environment
        self.model.load_state_dict(train_package["global_state_dict"])
        selected_malicious_clients = train_package["selected_malicious_clients"]
        server_round = train_package["server_round"]
        normalization = train_package.get("normalization", None)

        # Verify client is selected for poisoning
        assert self.client_id in selected_malicious_clients, "Client is not selected for poisoning"

        # Initialize poison attack (fine-tune Cerberus trigger and sync across clients if needed)
        if self.poison_module.sync_poison:  # Cerberus requires synchronization of trigger
            self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)

        # Normal training to get a reference parameter set
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - Normal training stage (reference params)")
        normal_model = copy.deepcopy(self.model)
        normal_model.load_state_dict(train_package["global_state_dict"])
        normal_model.train()

        normal_optimizer = instantiate(self.client_config.optimizer, params=normal_model.parameters())
        criterion = self.criterion

        # One local epoch of clean training on normal_model
        running_loss_clean = 0.0
        correct_clean = 0
        total_clean = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if isinstance(labels, torch.Tensor) and len(labels) <= 1:
                continue

            normal_optimizer.zero_grad()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if normalization:
                images = normalization(images)

            outputs = normal_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            normal_optimizer.step()

            running_loss_clean += loss.item() * len(labels)
            correct_clean += (outputs.argmax(dim=1) == labels).sum().item()
            total_clean += len(images)

        # Reference params from normal_model (detached)
        reference_params = {name: p.detach().clone() for name, p in normal_model.named_parameters()}

        # Poison training stage on self.model
        self.model.train()
        poison_epochs = self.atk_config.poison_epochs
        alpha_loss = float(getattr(self.atk_config, "alpha_loss", DEFAULT_PARAMS["alpha_loss"]))
        beta_loss = float(getattr(self.atk_config, "beta_loss", DEFAULT_PARAMS["beta_loss"]))  # not used here

        # Scheduler: follow reference MultiStep milestones based on poison_epochs if step_scheduler is enabled
        use_step_scheduler = bool(self.atk_config.get("step_scheduler", False))
        if use_step_scheduler:
            milestones = [
                max(1, int(0.2 * poison_epochs)),
                max(2, int(0.8 * poison_epochs))
            ]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=0.1
            )
        else:
            scheduler = None

        running_loss_bd = 0.0
        correct_bd = 0
        total_bd = 0

        for internal_epoch in range(poison_epochs):
            epoch_running_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if isinstance(labels, torch.Tensor) and len(labels) <= 1:
                    continue

                # Zero gradients
                self.optimizer.zero_grad()

                # Poison batch
                if self.atk_config.poison_mode == "multi_task":
                    # Clean and poisoned in a multi-task manner
                    clean_images = images.detach().clone()
                    clean_labels = labels.detach().clone()
                    poisoned_images = self.poison_module.poison_inputs(images)
                    poisoned_labels = self.poison_module.poison_labels(labels)

                    if normalization:
                        clean_images = normalization(clean_images)
                        poisoned_images = normalization(poisoned_images)

                    # Forward
                    clean_outputs = self.model(clean_images.to(self.device, non_blocking=True))
                    poisoned_outputs = self.model(poisoned_images.to(self.device, non_blocking=True))

                    clean_loss = criterion(clean_outputs, clean_labels.to(self.device, non_blocking=True))
                    poisoned_loss = criterion(poisoned_outputs, poisoned_labels.to(self.device, non_blocking=True))

                    # Combine losses
                    loss = (self.atk_config.attack_alpha * poisoned_loss +
                            (1 - self.atk_config.attack_alpha) * clean_loss)
                    outputs_for_acc = poisoned_outputs
                    labels_for_acc = poisoned_labels.to(self.device, non_blocking=True)

                elif self.atk_config.poison_mode in ["online", "offline"]:
                    if self.atk_config.poison_mode == "online":
                        images, labels = self.poison_module.poison_batch(batch=(images, labels))

                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if normalization:
                        images = normalization(images)

                    outputs_for_acc = self.model(images)
                    loss = criterion(outputs_for_acc, labels)
                    labels_for_acc = labels

                else:
                    raise ValueError(
                        f"Invalid poison_mode: {self.atk_config.poison_mode}. "
                        f"Expected one of: ['multi_task', 'online', 'offline']"
                    )

                # CerP distance loss to normal_model
                if alpha_loss > 0.0:
                    dist_loss = self._params_distance_norm(reference_params)
                    loss = loss + alpha_loss * dist_loss

                # (Optional) proximal term following protocol (FedProx-like)
                proximal_mu = train_package.get('proximal_mu', None) if self.atk_config.follow_protocol else None
                if proximal_mu is not None:
                    # Compute distance to global model (vectorized), using existing helper
                    global_params_tensor = torch.cat([
                        param.view(-1).detach().clone().requires_grad_(False)
                        for name, param in train_package["global_state_dict"].items()
                        if "weight" in name or "bias" in name
                    ]).to(self.device)
                    proximal_term = super().model_dist(global_params_tensor=global_params_tensor, gradient_calc=True)
                    loss = loss + (proximal_mu / 2) * proximal_term

                # Backward and step
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                epoch_running_loss += loss.item() * len(labels_for_acc)
                epoch_correct += (outputs_for_acc.argmax(dim=1) == labels_for_acc).sum().item()
                epoch_total += len(labels_for_acc)

            # Step scheduler if needed
            if scheduler is not None:
                scheduler.step()

            # Log epoch metrics (poison stage)
            epoch_loss = epoch_running_loss / max(1, epoch_total)
            epoch_acc = epoch_correct / max(1, epoch_total)
            if self.verbose:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                          f"- Poison Epoch {internal_epoch} | Train Loss: {epoch_loss:.4f} | "
                          f"Train Accuracy: {epoch_acc:.4f}")

            # Accumulate overall poison-stage metrics
            running_loss_bd += epoch_running_loss
            correct_bd += epoch_correct
            total_bd += epoch_total

        # Final metrics
        train_backdoor_loss = running_loss_bd / max(1, total_bd)
        train_backdoor_acc = correct_bd / max(1, total_bd)

        # Log final results
        log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} - "
                  f"Train Backdoor Loss: {train_backdoor_loss:.4f} | "
                  f"Train Backdoor Accuracy: {train_backdoor_acc:.4f} | "
                  )

        training_metrics = {
            "train_backdoor_loss": train_backdoor_loss,
            "train_backdoor_acc": train_backdoor_acc,
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