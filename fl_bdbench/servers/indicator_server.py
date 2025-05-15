"""
BackdoorIndicator server implementation for FL.
Reference: https://github.dev/ybdai7/Backdoor-indicator-defense/blob/main/participants/servers/IndicatorServer.py
"""

import torch
import numpy as np
import torchvision.transforms.v2 as transforms
import random
import copy
import torch.nn as nn
import os
import math
import time

from typing import List, Tuple, Dict, Any
from torchvision import datasets
from logging import INFO, WARNING
from fl_bdbench.servers.defense_categories import AnomalyDetectionServer
from fl_bdbench.utils.logging_utils import log
from fl_bdbench.const import client_id, StateDict, num_examples, Metrics

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
    "global_watermarking_start_round": 1000,
    "global_watermarking_end_round": 13000,
    "global_watermarking_round_interval": 1,
    "global_milestones": [
        10, 20, 30, 160, 200, 240, 280, 320, 360
    ],
    "global_lr_gamma": 0.8,  # 0.8 for green car
    "global_retrain_no_times": 200,
    "ood_data_sample_lens": 800,
    "ood_data_batch_size": 64,
    "ood_data_source": "CIFAR100",
    "watermarking_mu": 0.1,
    "replace_original_bn": True,
    "verbose": False,
    "VWM_detection_threshold": 95,
    "early_stopping": True
}

class IndicatorServer(AnomalyDetectionServer):
    """
    Indicator server that use OOD dataset to detect backdoor attacks.
    """

    defense_categories = ["anomaly_detection"]

    def __init__(self, server_config, server_type="indicator", eta: float = 0.1, **kwargs):
        super(IndicatorServer, self).__init__(server_config, server_type, eta)

        params_to_update = DEFAULT_SERVER_PARAMS.copy()
        params_to_update.update(kwargs)
        
        for key, value in params_to_update.items():
            setattr(self, key, value)

        self.watermarking_rounds = list(range(self.global_watermarking_start_round, self.global_watermarking_end_round, self.global_watermarking_round_interval))
        
        if self.config.dataset.upper() == "CIFAR10":
            self.ood_data_source = "CIFAR100"
        else:
            self.ood_data_source = "CIFAR10"

        self._get_ood_data()
        self.open_set = self._get_ood_dataloader()
        self.check_model = copy.deepcopy(self.global_model)

        assert self.ood_data_source.upper() != self.config["dataset"].upper(), "OOD data source must be different from training data source"
        self.after_wm_injection_bn_stats_dict = dict()

        log(INFO, f"Initialized Indicator server with watermarking_mu={self.watermarking_mu} and ood_data_source={self.ood_data_source}")

    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]):
        if self.current_round not in self.watermarking_rounds:
            malicious_clients = []
            benign_clients = [client_id for client_id, _, _ in client_updates]
            return malicious_clients, benign_clients
        
        benign_clients = []
        malicious_clients = []
        label_inds = []
        label_acc_ws = []
        
        # Batch process clients if possible
        for ind, (client_id, num_examples, model_state_dict) in enumerate(client_updates):            
            # Update only necessary parameters
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

            # Use cached batches
            total_l, watermark_acc, label_acc_w, label_ind, wm_label_acc_list, wm_label_dict = self._global_watermarking_test_sub(
                test_data=self.get_batches_iterator(), model=self.check_model)

            if self.verbose:
                log(INFO, f"client {client_id} | watermarking acc: {watermark_acc}, watermarking loss: {total_l}, target label ({label_ind}) wm acc: {label_acc_w}")
                log(INFO, wm_label_dict)
            
            label_inds.append(label_ind)
            label_acc_ws.append(label_acc_w)

            if label_acc_w < self.VWM_detection_threshold: 
                benign_clients.append(client_id)
            else:
                malicious_clients.append(client_id)
        
        log(INFO, f"label ind:{label_inds}")
        log(INFO, f"label acc wm:{label_acc_ws}") 
        return malicious_clients, benign_clients
    
    def fit_round(self, clients_mapping: Dict[Any, List[int]]) -> Metrics:
        """Perform one round of FL training. 
        
        Args:
            clients_mapping: Mapping of client types to list of client IDs
        Returns:
            aggregated_metrics: Dict of aggregated metrics from clients training
        """

        #### Server initialize by training on indicator task
        preprocess_time_start = time.time()
        self.pre_process(self.current_round) # Update global params
        preprocess_time_end = time.time()
        preprocess_time = preprocess_time_end - preprocess_time_start
        log(INFO, f"Indicator Server Initialization time: {preprocess_time:.2f} seconds")

        train_time_start = time.time()
        client_packages = self.trainer.train(clients_mapping)
        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        log(INFO, f"Clients training time: {train_time:.2f} seconds")

        client_metrics = []
        client_updates = []

        for client_id, package in client_packages.items():
            num_examples, model_updates, metrics = package
            client_metrics.append((client_id, num_examples, metrics))
            client_updates.append((client_id, num_examples, model_updates))

        aggregate_time_start = time.time()
            
        if self.aggregate_client_updates(client_updates):
            self.global_model.load_state_dict(self.global_model_params, strict=True)
            aggregated_metrics = self.aggregate_client_metrics(client_metrics)
        else:
            log(WARNING, "No client updates to aggregate. Global model parameters are not updated.")
            aggregated_metrics = {}
        
        aggregate_time_end = time.time()
        aggregate_time = aggregate_time_end - aggregate_time_start
        log(INFO, f"Server aggregate time: {aggregate_time:.2f} seconds")

        return aggregated_metrics
    
    def pre_process(self, round):
        wm_data = copy.deepcopy(self.open_set)
        log(INFO, f"Before indicator: ")
        loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
        log(INFO, f"watermarking acc: {acc_w}, watermarking loss: {loss_w}, target label ({label_ind}) wm acc: {label_acc_w}")

        metrics = self.server_evaluate(round_number=round)
        log(INFO, f"benign acc: {metrics['test_clean_acc']}, benign loss: {metrics['test_clean_loss']}")

        ### Initialize to calculate the distance between updates and global model
        if round in self.watermarking_rounds:
            log(INFO, f"Indicator Server: Perform training on indicator task..")
            target_params_variables = dict()
            for name, param in self.global_model.state_dict().items():
                target_params_variables[name] = param.clone()

            before_wm_injection_bn_stats_dict = dict()
            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    before_wm_injection_bn_stats_dict[key] = value.clone().detach()
            
            log(INFO, f"begin inserting new watermarking")
            wm_data = copy.deepcopy(self.open_set)
            self._global_watermark_injection(watermark_data=wm_data,
                            target_params_variables=target_params_variables,
                            model=self.global_model,
                            round=round
                            )

            watermarking_update_norm = self._model_dist_norm(self.global_model, target_params_variables)
            log(INFO, f"watermarking update norm is :{watermarking_update_norm}")

            wm_data = copy.deepcopy(self.open_set)

            log(INFO, f"After indicator: ")
            loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
            log(INFO, f"watermarking acc: {acc_w}, watermarking loss: {loss_w}, target label ({label_ind}) wm acc: {label_acc_w}")

            metrics = self.server_evaluate(round_number=round)
            log(INFO, f"benign acc: {metrics['test_clean_acc']}, benign loss: {metrics['test_clean_loss']}")

            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self._transfer_params(self.global_model, self.check_model)
            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])
                    if self.replace_original_bn:
                        self.global_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])

            log(INFO, f"after replace wm bn with original bn:")
            metrics = self.server_evaluate(round_number=round, model=self.check_model)
            log(INFO, f"benign acc: {metrics['test_clean_acc']}, benign loss: {metrics['test_clean_loss']}")

            ### Update global_model_params
            self.global_model_params = {name: param.detach().clone().to(self.device) for name, param in self.global_model.state_dict().items()}
        return True
    
    def _get_ood_data(self):
        """Get OOD data from the specified data source."""
        datapath = os.path.join(self.config["datapath"], self.ood_data_source)

        if self.ood_data_source == "CIFAR10":
            self.ood_dataset = datasets.CIFAR10(datapath, train=True, download=True, 
                                                transform=OOD_TRANSFORMATIONS["CIFAR10"])
        elif self.ood_data_source == "CIFAR100":
            self.ood_dataset = datasets.CIFAR100(datapath, train=True, download=True, 
                                                transform=OOD_TRANSFORMATIONS["CIFAR10"])
        elif self.ood_data_source == "EMNIST":
            self.ood_dataset = datasets.EMNIST(datapath, train=True, split="mnist", download=True,
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
        sample_index = random.sample(range(len(self.ood_dataset)), self.ood_data_sample_lens)
        for ind in sample_index:
            ood_data.append(self.ood_dataset[ind])
            assigned_label = random.randint(0, self.config["num_classes"] - 1)
            ood_data_label.append(assigned_label)
        return ood_data, ood_data_label

    def _get_ood_dataloader(self):
        """
        Sample limited OOD data as open set noise with balanced class distribution.
        Returns cached batches of OOD data with assigned labels.
        """
        # Ensure requested sample size doesn't exceed dataset size
        sample_size = min(self.ood_data_sample_lens, len(self.ood_dataset))
        batch_size = self.ood_data_batch_size
        
        # Sample indices without replacement
        indices = random.sample(range(len(self.ood_dataset)), sample_size)
        
        # Create dataloader with sampled indices
        ood_dataloader = torch.utils.data.DataLoader(
            self.ood_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            drop_last=True,
            num_workers=4,
            pin_memory=True
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
        
        # Cache processed batches in memory to avoid repeated dataloader iterations
        batches = []
        for batch_id, (data, _) in enumerate(ood_dataloader):
            # Add channel dimension for grayscale images                
            if "NIST" in self.ood_data_source: 
                data = data.repeat(1, 3, 1, 1)
                
            # Assign new labels and move to GPU once
            targets = torch.tensor(assigned_labels[batch_id])
            data = data.cuda().requires_grad_(False)
            targets = targets.cuda().requires_grad_(False)
            batches.append((data, targets))
        
        return batches

    def get_batches_iterator(self):
        """Return an iterator over the cached batches."""
        return iter(self.open_set)
    
    def _loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
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
    
    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.global_model, target_params_variables)
        if model_norm > self.global_projection_norm and self.global_is_projection_grad:
            norm_scale = self.global_projection_norm / model_norm
            for name, param in self.global_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)
        return True

    def _global_watermark_injection(self, watermark_data, target_params_variables, round=None, model=None):
        if model is None:
            model = self.global_model
        model.train()

        self._loss_function()
        self._optimizer(round, model)
        self._scheduler()

        log(INFO, f"watermarking_mu:{self.watermarking_mu}")
        
        # Use cached data directly
        batches = watermark_data
        retrain_no_times = self.global_retrain_no_times
        
        # Early stopping if loss converges
        prev_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for internal_round in range(retrain_no_times):
            round_loss = 0
            batch_count = 0
            
            if internal_round % 50 == 0:
                log(INFO, f"global watermarking injection round: {internal_round}")
            
            for data, targets in batches:
                self.optimizer.zero_grad()
                
                # Data is already on GPU from caching
                output = model(data)
                
                class_loss = self.criterion(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                loss = class_loss + (self.watermarking_mu/2) * distance_loss
                
                loss.backward()
                self.optimizer.step()
                
                self._projection(target_params_variables)
                round_loss += loss.item()
                batch_count += 1
            
            avg_round_loss = round_loss / batch_count
            
            # Early stopping check
            if self.early_stopping:
                if abs(prev_loss - avg_round_loss) < 1e-3:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log(INFO, f"Early stopping at round {internal_round}/{retrain_no_times}")
                        break
                else:
                    patience_counter = 0
            
            prev_loss = avg_round_loss
            self.scheduler.step()
            
            # Evaluate less frequently
            if internal_round == retrain_no_times-1 or internal_round % 50 == 0:
                metrics = self.server_evaluate(round_number=round, test_poisoned=False, model=model)
                log(INFO, f"round: {internal_round} | benign acc:{metrics['test_clean_acc']}, benign loss:{metrics['test_clean_loss']}")
                
                loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(
                    test_data=self.get_batches_iterator(), model=model)
                log(INFO, f"watermarking acc: {acc_w}, watermarking loss: {loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

        return True

    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)
    
    def _model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        
        # Replace deprecated torch.cuda.FloatTensor with recommended approach
        sum_var = torch.zeros(size, dtype=torch.float32, device=self.device)
        
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)
        
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
            total_loss += torch.nn.functional.cross_entropy(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            if batch_id==0 and model != None and self.verbose:
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
    
    def _transfer_params(self, source_model, target_model):
        source_params = source_model.state_dict()
        target_params = target_model.state_dict()
        for name, param in source_params.items():
            if name in target_params:
                target_params[name].copy_(param.clone())
