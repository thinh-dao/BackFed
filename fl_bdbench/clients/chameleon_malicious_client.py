"""
Chameleon client implementation for FL.
"""

import torch
import torch.nn as nn
import math
import copy

from torch.nn import functional as F
from omegaconf import open_dict
from torchvision.models import VGG, ResNet
from fl_bdbench.clients.base_malicious_client import MaliciousClient
from fl_bdbench.utils import log 
from logging import INFO

DEFAULT_PARAMS = {
    "poisoned_supcon_retrain_no_times": 10,
    "poisoned_supcon_lr": 0.015,
    "poisoned_supcon_momentum": 0.9,
    "poisoned_supcon_weight_decay": 0.0005,
    "poisoned_supcon_milestones": [2, 4, 6, 8],
    "poisoned_supcon_lr_gamma": 0.3,
    "fac_scale_weight": 2,
}

class ChameleonClient(MaliciousClient):
    """
    Chameleon client implementation for FL.
    """

    def __init__(self, client_id, dataset, dataset_indices, model, client_config, atk_config, poison_module, context_actor, **kwargs):
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
            client_type="chameleon",
            **params_to_update
        )
        
    def train_contrastive_model(self, train_package):
        """
        Train the model maliciously for a number of epochs.
        
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        """
        self.contrastive_model = copy.deepcopy(self.model)
        self.contrastive_model.train()

        # Modify the model to be a SupConModel
        if isinstance(self.contrastive_model, VGG):
            self.contrastive_model.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.contrastive_model.classifier = nn.Identity()
        elif isinstance(self.contrastive_model, ResNet):
            self.contrastive_model.fc = nn.Identity()

        self._loss_function()
        self._supcon_optimizer()
        self._supcon_scheduler()

        data_iterator = self.train_loader
        target_params_variables = self.model.state_dict()

        for internal_round in range(self.atk_config["poisoned_supcon_retrain_no_times"]):
            for batch_idx, batch in enumerate(data_iterator):
                self.supcon_optimizer.zero_grad()
                batch = self.poison.poison_batch(batch, mode="train")
                data, targets = batch
                if train_package.get("normalization", None):
                    data = train_package["normalization"](data)
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = F.normalize(self.contrastive_model(data), dim=1)

                contrastive_loss = self.supcon_loss(output, targets,
                                                    scale_weight=self.atk_config["fac_scale_weight"],
                                                    fac_label=self.atk_config["target_class"])
                if train_package.get("proximal_mu", None):
                    distance_loss = self._model_dist_norm(self.contrastive_model, target_params_variables)
                    loss = contrastive_loss + (train_package["proximal_mu"]/2) * distance_loss
                else:
                    loss = contrastive_loss
                
                loss.backward()
                self.supcon_optimizer.step()
                
                if self.atk_config["poisoned_is_projection"] and \
                    (batch_idx + 1) % self.atk_config["poisoned_projection_frequency"] == 0 and \
                    batch_idx == len(data_iterator) - 1:
                    self._projection(self.contrastive_model, target_params_variables)

            self.supcon_scheduler.step()
            backdoor_loss, backdoor_accuracy = self.poison.poison_test(self.model, self.train_loader)
            
            if self.verbose and self.atk_config["poisoned_supcon_retrain_no_times"] % (self.atk_config["poisoned_supcon_retrain_no_times"] // 5) == 0:
                log(INFO, f"Client [{self.client_id}] ({self.client_type}) at round {train_package['server_round']} - Epoch {internal_round} | Contrastive loss: {contrastive_loss.item()} | Backdoor loss: {backdoor_loss} | Backdoor accuracy: {backdoor_accuracy}")

        # Transfer the trained weights of encoder to the local model and freeze the encoder
        self.transfer_params(source_model=self.contrastive_model, target_model=self.model)
        for params in self.model.named_parameters():
            if "linear.weight" not in params[0] and "linear.bias" not in params[0]:
                params[1].require_grad = False

        return True
    
    def train(self, train_package):
        """
        Train the model maliciously.
        """
        self.train_contrastive_model(train_package.get("normalization", None))
        super().train(train_package)
    
    def _loss_function(self):
        self.supcon_loss = SupConLoss().cuda()
        return True

    def _supcon_optimizer(self): 
        self.supcon_optimizer = torch.optim.SGD(self.contrastive_model.parameters(), lr=self.atk_config["poisoned_supcon_lr"],
                                    momentum=self.atk_config["poisoned_supcon_momentum"], weight_decay=self.atk_config["poisoned_supcon_weight_decay"])  
        return True
    
    def _supcon_scheduler(self):
        self.supcon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.supcon_optimizer,
                                                 milestones=self.atk_config['poisoned_supcon_milestones'],
                                                 gamma=self.atk_config['poisoned_supcon_lr_gamma'])
        return True

    def _model_dist_norm_var(self, model, target_params_variables, ord=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.zeros(size, device="cuda")
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.linalg.norm(sum_var, ord=ord)

    def _model_dist_norm(self, model, target_params_variables):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params_variables[name].data, 2))
        return math.sqrt(squared_sum)
    
    def _projection(self, model, target_params_variables):
        model_norm = self._model_dist_norm_var(model, target_params_variables)

        if model_norm > self.atk_config["poisoned_projection_eps"]:
            norm_scale = self.atk_config["poisoned_projection_eps"] / model_norm
            for name, param in model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True   

    def transfer_params(self, source_model, target_model):
        source_params = source_model.state_dict()
        target_params = target_model.state_dict()
        for name, param in source_params.items():
            if name in target_params:
                target_params[name].copy_(param.clone())

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


