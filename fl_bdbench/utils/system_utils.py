"""
System utility functions for FL.
"""

import os
import random
import socket
import numpy as np
import torch
import ray

from typing import Dict, Union
from datetime import datetime
from omegaconf import DictConfig
from logging import INFO, WARNING
from fl_bdbench.const import NUM_CLASSES
from fl_bdbench.utils.logging_utils import log

def system_startup(config: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    total_cpus = os.cpu_count()
    total_gpus = torch.cuda.device_count()
    client_cpus = config.num_cpus
    client_gpus = config.num_gpus
    num_parallel = min(int(total_cpus / client_cpus), int(total_gpus / client_gpus))

    if config.mode == "parallel":
        namespace = f"{config.dataset}_{config.aggregator}"
        ray_init(num_gpus=total_gpus, num_cpus=total_cpus, namespace=namespace)
        
    # Log system information
    log(INFO, f"Date: {datetime.now().strftime('%Y-%m-%d')} | PyTorch {torch.__version__}") 
    log(INFO, f'Total CPUs: {total_cpus}, Total GPUs: {total_gpus} on {socket.gethostname()}.')
    log(INFO, f"Client CPUS: {config.num_cpus}, Client GPUs: {config.num_gpus}")

    if config.mode == "parallel":
        log(INFO, f"Number of concurrent clients: {num_parallel}")
    
    # Set random seed
    set_random_seed(config.seed, config.deterministic)
    log(INFO, f"Set random seed to {config.seed}")
    
    # Switch to debug settings
    if config.debug:
        log(INFO, "Debug mode enabled, adjusting settings")
        set_debug_settings(config)
            
    # Set fraction_evaluation to 0 if val_split = 0 (since there is no validation set)
    if config["federated_val_split"] == 0.0 and config["federated_evaluation"] == True:
        config["federated_evaluation"] = False
        log(WARNING, "Setting federated_evaluation to False since federated_val_split = 0 (no validation set from clients)")

    # Override basic config
    config.num_classes = NUM_CLASSES[config.dataset.upper()]
    config.client_config.optimizer = config.client_optimizer_config[config.client_config.optimizer] # Now store DictConfig in client_optimizer_config
    config.atk_config.atk_optimizer = config.atk_config.atk_optimizer_config[config.atk_config.atk_optimizer] # Now store DictConfig in atk_optimizer_config
    
    # Set attack config
    if config.no_attack == False:
        set_attack_config(config)

def ray_init(num_gpus: int, num_cpus: int, namespace: str):
    try:
        ray.init(
            namespace=namespace,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            ignore_reinit_error=True,
        )
    except ValueError:
        # If Ray is already running, connect to it
        ray.init(
            address="auto",
            namespace=namespace,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            ignore_reinit_error=True,
        )

def set_attack_config(config: DictConfig):    
    # Define malicious clients if random_adversaries is True
    if config.atk_config.random_adversaries:
        num_adversaries = int(config.atk_config.fraction_adversaries * config.num_clients)
        config.atk_config.malicious_clients = random.sample(range(config.num_clients), num_adversaries)

    log(INFO, f"Malicious clients: {config.atk_config.malicious_clients}")
    
    # Define target and source class if random_class is True
    if config.atk_config.random_class:
        config.atk_config.target_class = random.randint(0, config.num_classes - 1)
        config.atk_config.source_class = random.randint(0, config.num_classes - 1)
        while config.atk_config.source_class == config.atk_config.target_class:
            config.atk_config.source_class = random.randint(0, config.num_classes - 1)
    
    if config.atk_config.attack_type == "all2one":
        log(INFO, f"Attack type: {config.atk_config.attack_type}, Target class: {config.atk_config.target_class}, Source class: {config.atk_config.source_class}")
    else:
        log(INFO, f"Attack type: {config.atk_config.attack_type}, Target class: {config.atk_config.target_class}")

def set_random_seed(seed=123123, deterministic=False):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_debug_settings(config):
    """Set config for debug settings"""

    # FL settings
    config.num_rounds = 4
    config.num_clients = 4
    config.num_gpus = 0.5
    config.train_batch_size = 128
    config.simulation.fraction_fit = 1
    config.simulation.min_fit_clients = 4
    config.simulation.min_evaluate_clients = 4
    config.save_logging = None
    
    # Attacker settings
    config.atk_config.fraction_adversaries = 0.25
    config.atk_config.poison_start_round = 1
    config.atk_config.poison_end_round = 2
    config.atk_config.poison_interval = 1
    
    # Mofiy poison_start and poison_end round if load from checkpoint
    # Currently not supported for wandb checkpoint
    if isinstance(config.checkpoint, int):
        config.atk_config.poison_start_round = config.checkpoint + 1
        config.atk_config.poison_end_round = config.checkpoint + 2

    if isinstance(config.checkpoint, str):
        if config.checkpoint.split("_")[-1][:-4] == "uniform":
            round_num = int(config.checkpoint.split("_")[-2])
        else:
            round_num = int(config.checkpoint.split("_")[-3])
        config.atk_config.poison_start_round = round_num + 1
        config.atk_config.poison_end_round = round_num + 2

    if config.atk_config.mutual_dataset == True:
        config.atk_config.num_attacker_samples = 100

def pool_size_from_resources(client_resources: Dict[str, Union[int, float]]) -> int:
    """Calculate maximum number of actors that can fit in the cluster based on resources."""
    total_actors = 0
    client_cpus = client_resources["num_cpus"]
    client_gpus = client_resources.get("num_gpus", 0.0)

    for node in ray.nodes():
        resources = node.get("Resources", {})
        if not resources:
            continue
            
        node_actors = int(resources["CPU"] / client_cpus)
        
        if client_gpus > 0:
            node_gpus = resources.get("GPU", 0)
            node_actors = min(node_actors, int(node_gpus / client_gpus)) if node_gpus else 0
            
        total_actors += node_actors

    if total_actors == 0:
        raise ValueError(
            f"Cannot create actor pool: insufficient resources. "
            f"Required per client: {client_resources}"
        )

    return total_actors
