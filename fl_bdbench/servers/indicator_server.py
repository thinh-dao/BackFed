"""
BackdoorIndicator server implementation for FL.
Reference: https://github.dev/ybdai7/Backdoor-indicator-defense/blob/main/participants/servers/IndicatorServer.py
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import random
import copy
import torch.nn as nn

from typing import List, Tuple
from torchvision import datasets
from logging import INFO
from fl_bdbench.servers.defense_categories import AnomalyDetectionServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import client_id, StateDict, num_examples

OOD_TRANSFORMATIONS = {
    "EMNIST": transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]),
    "CIFAR10": transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ]),
}

DEFAULT_SERVER_PARAMS = {
    "global_lr": 0.005,  # 0.001
    "global_momentum": 0.9,  # 0.9
    "global_weight_decay": 0.0005,
    "global_is_projection_grad": False,
    "global_projection_norm": 0.8,
    "global_watermarking_start_round": 1100,
    "global_watermarking_end_round": 13000,
    "global_watermarking_round_interval": 1,
    "global_milestones": [
        10, 20, 30, 160, 200, 240, 280, 320, 360
    ],
    "global_lr_gamma": 0.8,  # 0.8 for green car
    "global_retrain_no_times": 200,
    "global_ood_data_sample_lens": 800,
    "global_ood_data_batch_size": 64,
    "global_ood_data_source": "CIFAR10",
    "global_ood_data_transformations": OOD_TRANSFORMATIONS["CIFAR10"],
    "watermarking_mu": 0.1,
    "ood_data_source": "CIFAR10",
    "ood_batch_size": 64,
    "ood_data_sample_lens": 800,
    "replace_original_bn": True,
}

class IndicatorServer(AnomalyDetectionServer):
    """
    Indicator server that use OOD dataset to detect backdoor attacks.
    """

    defense_categories = ["anomaly_detection"]

    def __init__(self, server_config, server_type="indicator",**kwargs):
        super(IndicatorServer, self).__init__(server_config, server_type)

        params_to_update = DEFAULT_SERVER_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)

        self.watermarking_rounds = list(range(self.global_watermarking_start_round, self.global_watermarking_end_round, self.global_watermarking_round_interval))
        self._get_ood_data()
        self.open_set = self._get_ood_dataloader()

        assert self.ood_data_source.upper() != self.config["dataset"].upper(), "OOD data source must be different from training data source"
        self.after_wm_injection_bn_stats_dict = dict()

        log(INFO, f"Initialized Indicator server with watermarking_mu={self.watermarking_mu} and ood_data_source={self.ood_data_source}")

    def pre_process(self, round):
        wm_data = copy.deepcopy(self.open_set)
        loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
        log(INFO, f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

        self.server_evaluate(round=round)
        log(INFO, f" ")

        ### Initialize to calculate the distance between updates and global model
        if round in self.watermarking_rounds:
            target_params_variables = dict()
            for name, param in self.global_model.state_dict().items():
                target_params_variables[name] = param.clone()

            before_wm_injection_bn_stats_dict = dict()
            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    before_wm_injection_bn_stats_dict[key] = value.clone().detach()
            
            log(INFO, f"benign inserting new watermarking")
            wm_data = copy.deepcopy(self.open_set)
            self._global_watermark_injection(watermark_data=wm_data,
                            target_params_variables=target_params_variables,
                            model=self.global_model,
                            round=round
                            )

            watermarking_update_norm = self._model_dist_norm(self.global_model, target_params_variables)
            log(INFO, f"watermarking update norm is :{watermarking_update_norm}")

            wm_data = copy.deepcopy(self.open_set)
            loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
            log(INFO, f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

            self.server_evaluate(round=round)

            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self.check_model.copy_params(self.global_model.state_dict())
            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key].\
                        copy_(before_wm_injection_bn_stats_dict[key])
                    if self.replace_original_bn:
                        self.global_model.state_dict()[key].\
                            copy_(before_wm_injection_bn_stats_dict[key])

            log(INFO, f"after replace wm bn with original bn:")
            self.server_evaluate(round=round, model=self.check_model)

            log(INFO, f" ")
        return True
    
    def _get_ood_data(self):
        """Get OOD data from the specified data source."""
        if self.ood_data_source == "CIFAR10":
            self.ood_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                transform=OOD_TRANSFORMATIONS["CIFAR10"])
        elif self.ood_data_source == "CIFAR100":
            self.ood_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                transform=OOD_TRANSFORMATIONS["CIFAR10"])
        elif self.ood_data_source == "EMNIST":
            self.ood_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                        transform=OOD_TRANSFORMATIONS["EMNIST"])
        else:
            raise ValueError(f"OOD data source {self.ood_data_source} is not supported.")

        return True

    def _get_sample(self):
        """
        Sample limited ood data as open set noise
        """
        ood_data = list()
        ood_data_label = list()
        sample_index = random.sample(range(len(self.ood_dataset)), self.global_ood_data_sample_lens)
        for ind in sample_index:
            ood_data.append(self.ood_dataset[ind])
            assigned_label = random.randint(0, self.config["num_classes"] - 1)
            ood_data_label.append(assigned_label)
        return ood_data, ood_data_label

    def _get_ood_dataloader(self):
        """
        Sample limited OOD data as open set noise with balanced class distribution.
        Returns an iterator over batches of OOD data with assigned labels.
        """
        # Ensure requested sample size doesn't exceed dataset size
        sample_size = min(self.global_ood_data_sample_lens, len(self.ood_dataset))
        batch_size = self.global_ood_data_batch_size
        
        # Sample indices without replacement
        indices = random.sample(range(len(self.ood_dataset)), sample_size)
        
        # Create dataloader with sampled indices
        ood_dataloader = torch.utils.data.DataLoader(
            self.ood_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            drop_last=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True  # Faster data transfer to GPU
        )
        
        # Calculate actual number of samples after dropping incomplete batches
        num_batches = len(ood_dataloader)
        actual_sample_size = num_batches * batch_size
        
        # Create balanced class distribution for assigned labels
        num_classes = self.config["num_classes"]
        repeats = actual_sample_size // num_classes
        remainder = actual_sample_size % num_classes
        
        # Create balanced labels array and reshape by batch
        assigned_labels = np.concatenate([
            np.repeat(np.arange(num_classes), repeats),
            np.arange(remainder)
        ])
        np.random.shuffle(assigned_labels)
        assigned_labels = assigned_labels.reshape(num_batches, batch_size)
        
        # Process each batch
        processed_batches = []
        for batch_id, (data, targets) in enumerate(ood_dataloader):
            # Handle EMNIST dataset (grayscale to RGB conversion if needed)
            if self.config["dataset"].upper() == "EMNIST":
                data = data[:, 0, :, :].unsqueeze(1)  # Keep only first channel
                
            if self.ood_data_source == "EMNIST":
                data = data.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
                
            # Assign new labels
            targets = torch.tensor(assigned_labels[batch_id])
            processed_batches.append((data, targets))
        
        return iter(processed_batches)
    
    def _loss_function(self):
        self.ceriterion = self.ceriterion_build
        return True

    def _optimizer(self, round, model):
        lr = self.global_lr
        momentum = self.global_momentum 
        weight_decay = self.global_weight_decay 

        log(INFO, f"indicator lr:{lr}")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        return True

    def _scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.global_milestones,
                                                 gamma=self.global_lr_gamma)
        return True
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]):
        if self.current_round not in self.watermarking_rounds:
            malicious_clients = []
            benign_clients = [client_id for client_id, _, _ in client_updates]
            return malicious_clients, benign_clients
        
        self.pre_process(round=self.current_round)
        benign_clients = []
        label_inds = []
        label_acc_ws = []

        for ind, (client_id, num_examples, model_state_dict) in enumerate(client_updates):

            self.check_model.copy_params(self.global_model.state_dict())
            for name, data in model_state_dict.items():
                if "num_batches_tracked" in name:
                    continue

                if "running" in name:
                    if self.replace_original_bn:
                        new_value = self.after_wm_injection_bn_stats_dict[name]
                    else:
                        continue
                else:
                    new_value = data.clone().detach()

                self.check_model.state_dict()[name].copy_(new_value)

            wm_copy_data = copy.deepcopy(self.open_set)
            _, _, label_acc_w, label_ind, _, _ \
                = self._global_watermarking_test_sub(test_data=wm_copy_data, model=self.check_model)
            
            label_inds.append(label_ind)
            label_acc_ws.append(label_acc_w)

            if label_acc_w < self.VWM_detection_threshold: 
                benign_clients.append(client_id)
            else:
                malicious_clients.append(client_id)
        
        log(INFO, f"label ind:{label_inds}")
        log(INFO, f"label acc wm:{label_acc_ws}") 
        return benign_clients, malicious_clients

    def _global_watermark_injection(self, watermark_data, target_params_variables, round=None, model=None):

        if model==None:
            model = self.global_model
        model.train()

        total_loss = 0
        self._loss_function()
        self._optimizer(round, model)
        self._scheduler()

        log(INFO, f"wm_mu:{self.wm_mu}")

        retrain_no_times = self.global_retrain_no_times
        
        for internal_round in range(retrain_no_times):

            if internal_round%50==0:
                log(INFO, f"global watermarking injection round:{internal_round}")
            data_iterator = copy.deepcopy(watermark_data)

            for batch_id, watermark_batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                wm_data, wm_targets = watermark_batch                
                wm_data = wm_data.cuda().detach().requires_grad_(False)
                wm_targets = wm_targets.cuda().detach().requires_grad_(False)

                data = wm_data
                targets = wm_targets

                output = model(data) 
                pred = output.data.max(1)[1]

                class_loss = nn.functional.cross_entropy(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                loss = class_loss + (self.wm_mu/2) * distance_loss 

                loss.backward()
                self.optimizer.step()
                
                self._projection(target_params_variables)
                total_loss += loss.data

                if internal_round == retrain_no_times-1 and batch_id==0:
                    metrics = self.server_evaluate(test_poisoned=False, model=model)
                    log(INFO, f"round:{internal_round} | benign acc:{metrics['test_clean_acc']}, benign loss:{metrics['test_clean_loss']}")

                    wm_data = copy.deepcopy(self.open_set)
                    loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=model)
                    log(INFO, f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

                    log(INFO, f" ")

            self.scheduler.step()

        return True
    
    def _global_watermarking_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        wm_label_correct = 0
        wm_label_sum = 0
        data_iterator = test_data

        wm_label_sum_list = [0 for i in range(self.config["num_classes"])]
        wm_label_correct_list = [0 for i in range(self.config["num_classes"])]
        wm_label_acc_list = [0 for i in range(self.config["num_classes"])]
        wm_label_dict = dict()
        for i in range(self.config["num_classes"]):
            wm_label_dict[i] = 0

        for batch_id, batch in enumerate(data_iterator):

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            if batch_id==0 and model != None and self.show_train_log:
                log(INFO, f"watermarking targets:{targets}")
                log(INFO, f"watermarking pred :{pred}")
            
            for pred_item in pred:
                wm_label_dict[pred_item.item()]+=1

            # poisoned_label = self.params["poison_label_swap"]
            for target_label in range(self.config["num_classes"]):
                wm_label_targets = torch.ones_like(targets) * target_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))

                wm_label_sum_list[target_label] += wm_label_index.cpu().sum().item()
                wm_label_correct_list[target_label] += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item() 

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)
            
        watermark_acc = 100.0 *(float(correct) / float(dataset_size))
        for i in range(self.config["num_classes"]):
            wm_label_dict[i] = round(wm_label_dict[i]/dataset_size,2)
        for target_label in range(self.config["num_classes"]):
            wm_label_acc_list[target_label] = round(100.0 * (float(wm_label_correct_list[target_label]) / float(wm_label_sum_list[target_label])), 2)

        # wm_label_acc = 100.0 * (float(wm_label_correct) / float(wm_label_sum))
        wm_label_acc = max(wm_label_acc_list)
        wm_index_label = wm_label_acc_list.index(wm_label_acc)
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, watermark_acc, wm_label_acc, wm_index_label, wm_label_acc_list, wm_label_dict)
