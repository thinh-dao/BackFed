from abc import ABC
from omegaconf import DictConfig
from typing import Any

import torch
import asyncio


class Poison(ABC):
    def __init__(self, params: DictConfig, client_id: int = -1, **kwargs):
        """
        Initialize the poison module.

        Args:
            params (DictConfig): Attack configuration
        """
        self.params = params
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def poison_batch(self, batch, mode="train"):
        poison_inputs, poison_labels = batch
        poison_inputs = poison_inputs.to(self.device, non_blocking=True)
        poison_labels = poison_labels.to(self.device, non_blocking=True)

        filter_mask = self.get_filter_mask(poison_labels, mode)
        poison_inputs[filter_mask] = self.poison_inputs(poison_inputs[filter_mask])
        poison_labels[filter_mask] = self.poison_labels(poison_labels[filter_mask])

        if mode == "train":
            return poison_inputs, poison_labels
        elif mode == "test":
            return poison_inputs[filter_mask], poison_labels[filter_mask]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def poison_warmup(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """
        Not required for all attacks. Called at the start of the poisoning round to update resources (trigger pattern, trigger generator, etc.).
        Could be used for training the trigger pattern, attacker model, etc.

        Args:
            client_id (int): The client id that update the resource
            server_round (int): The server round
            initial_model (torch.nn.Module): The initial model
            dataloader (torch.utils.data.DataLoader): The dataloader
            normalization (torch.nn.Module): The normalization
        """
        pass

    def poison_inputs(self, inputs, mode, *args, **kwargs):
        """
        Return the poisoned inputs (inputs with the trigger applied).
        'mode' argument is given if poison inputs during training and testing may be different. Otherwise, this should not matter.
        Args:
            inputs (torch.Tensor): Inputs to poison
            labels (torch.Tensor): Labels to poison
            test (bool): Whether to poison the entire batch or a portion of it
        Return:
            poisoned_inputs (torch.Tensor): Poisoned inputs
        """
        pass

    def poison_finish(self):
        """
        Not required for all attacks. Called at the end of the experiment.
        Could be used to delete the trigger, attacker model, etc.
        """
        pass

    def poison_labels(self, labels):
        """
        Return the poisoned labels.
        Args:
            labels (torch.Tensor or int): Labels to poison. Can be a tensor or a single integer.
            source_target_mappings (dict): Source-target mappings for the labels
            test (bool): Whether to poison the entire batch or a portion of it
        Return:
            poisoned_labels (torch.Tensor or int): Poisoned labels in the same format as input
        """
        # Handle scalar input (int or 0-dim tensor)
        is_scalar = isinstance(labels, int) or (torch.is_tensor(labels) and labels.dim() == 0)

        if is_scalar:
            # Handle scalar inputs directly
            if self.params.attack_type == "all2all":
                return (labels + 1) % self.params.num_classes
            elif self.params.attack_type == "all2one":
                return self.params.target_class
            elif self.params.attack_type == "one2one":
                return self.params.target_class if labels == self.params.source_class else labels
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")

        # Handle tensor inputs
        if self.params.attack_type == "all2all":
            return (labels + 1) % self.params.num_classes
        elif self.params.attack_type == "all2one":
            return torch.ones(len(labels), dtype=torch.long, device=self.device) * self.params.target_class
        elif self.params.attack_type == "one2one":
            return torch.where(labels == self.params.source_class,
                            torch.tensor(self.params.target_class, dtype=torch.long, device=self.device),
                            labels)
        else:
            raise ValueError(f"Invalid attack_type: {self.params.attack_type}")

    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """Validate attack success rate. We inject the trigger in samples from the source classes (excluding target classes)
        and then test the model on the poisoned samples.

        Args:
            net (torch.nn.Module): The model to test
            test_loader (torch.utils.data.DataLoader): The test loader
            loss_fn (torch.nn.Module): The loss function to use
            normalization (torch.utils.transforms.Normalize): The normalization

        Returns:
            backdoor_loss (float): The loss of backdoor target samples
            backdoor_accuracy (float): The accuracy of targeted misclassification
        """
        net.eval()
        backdoored_preds, total_samples, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for batch in test_loader:
                poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

                if normalization:
                    poisoned_inputs = normalization(poisoned_inputs)

                outputs = net(poisoned_inputs)
                backdoored_preds += (torch.max(outputs.data, 1)[1] == poisoned_labels).sum().item()
                total_loss += loss_fn(outputs, poisoned_labels).item()
                total_samples += len(poisoned_labels)

        backdoor_accuracy = backdoored_preds / total_samples
        backdoor_loss = total_loss / len(test_loader)
        return backdoor_loss, backdoor_accuracy

    def get_filter_mask(self, labels, mode):
        """Filter mask for samples in mask. Only the masked samples are triggered and evaluated in poison_test"""
        if mode == "train":
            num_poisons = int(self.params.poison_rate * len(labels))
            if self.params.attack_type == "all2all" or self.params.attack_type == "all2one":
                filter_mask = torch.arange(len(labels)) < num_poisons
            elif self.params.attack_type == "one2one":
                filter_mask = torch.isin(labels[:num_poisons], torch.tensor([self.params.source_class, self.params.target_class]))
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")
        elif mode == "test":
            if self.params.attack_type == "all2all":
                filter_mask = torch.ones(len(labels), dtype=torch.bool)
            elif self.params.attack_type == "one2one":
                filter_mask = torch.where(labels == self.params.source_class, True, False)
            elif self.params.attack_type == "all2one":
                filter_mask = torch.where(labels != self.params.target_class, True, False)
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return filter_mask

    def set_client_id(self, client_id: int):
        self.client_id = client_id
    
    def set_device(self, device: torch.device):
        self.device = device

    def __repr__(self) -> str:
        return self.__class__.__name__

