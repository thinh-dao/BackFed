"""
Tracking and visualization utilities for FL.
"""

import os
import re
import csv
import torch
import wandb
import tempfile
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from omegaconf import OmegaConf
from logging import INFO
from fl_bdbench.utils.logging_utils import log, CSVLogger

def init_wandb(config):
    # Login to wandb
    load_dotenv()        
    api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=api_key)

    # Define name for wandb run
    aggregator = config.aggregator

    if config.no_attack:
        attack_name = "noattack"
    else:
        attack_name = f"{config.atk_config.model_poison_method}({config.atk_config.data_poison_method})"

    if config.partitioner.lower() == "dirichlet":
        partitoner = f"dirichlet({config.alpha})"
    else:
        partitoner = "uniform"

    config.wandb.name = f"{attack_name}_{aggregator.lower()}_{config.dataset.lower()}_{partitoner}_{config.atk_config.selection_scheme}_{config.atk_config.poison_scheme}"
    if config.name_tag:
        config.wandb.name = f"{config.wandb.name}_{config.name_tag}"

    # Check if the run already exists and increment the version if it does
    api = wandb.Api()

    # Escape parentheses in the regex pattern
    escaped_name = re.escape(config.wandb.name)

    # Use the escaped name in the filter
    runs = api.runs(
        path=f"{config.wandb.entity}/{config.wandb.project}",
        filters={"display_name": {"$regex": f"^{escaped_name}.*"}}
    )
    config.wandb.name = f"{config.wandb.name}_v{len(runs)}"

    # Initialize wandb and define step metric
    wandb.init(
        project=config.wandb.project, 
        name=config.wandb.name, mode=config.wandb.mode, 
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )
    wandb.define_metric("*", step_metric="round")

def init_csv_logger(config, resume=False, validation=False):
    if config.partitioner.lower() == "dirichlet":
        partitoner = f"dirichlet({config.alpha})"
    else:
        partitoner = "uniform"

    aggregator = config.aggregator

    if config.no_attack:
        attack_name = "noattack"
    else:
        attack_name = f"{config.atk_config.model_poison_method}({config.atk_config.data_poison_method})"

    name = f"{attack_name}_{aggregator.lower()}_{config.dataset.lower()}_{partitoner}_{config.atk_config.selection_scheme}_{config.atk_config.poison_scheme}"
    if config.name_tag:
        name = f"{name}_{config.name_tag}"

    if config.dir_tag:
        dir_path = os.path.join("csv_results", config.dir_tag)
    else:
        dir_path = "csv_results"

    # Check if the run already exists and increment the version if it does
    os.makedirs(dir_path, exist_ok=True)
    count = 0
    for csv_file in os.listdir(dir_path):
        if name in csv_file:
            count += 1
    name = f"{name}_v{count}"

    file_name = os.path.join(dir_path, f"{name}.csv")
    field_names = ["round", "test_clean_loss", "test_clean_acc", "test_backdoor_loss", "test_backdoor_acc", "train_clean_loss", "train_clean_acc", "train_backdoor_loss", "train_backdoor_acc"]
    if validation:
        field_names.extend(["val_clean_loss", "val_clean_acc", "val_backdoor_loss", "val_backdoor_acc"])

    csv_logger = CSVLogger(fieldnames=field_names, resume=resume, filename=file_name)
    return csv_logger

def plot_csv(csv_path, fig_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        dict_of_lists = {}
        ks = None
        for i, r in enumerate(reader):
            if i == 0:
                for k in r:
                    dict_of_lists[k] = []
                ks = r
            else:
                for _i, v in enumerate(r):
                    if v == '':
                        break
                    dict_of_lists[ks[_i]].append(float(v))
    fig = plt.figure()
    for k in dict_of_lists:
        if k == 'round' or len(dict_of_lists[k])==0:
            continue
        plt.clf()
        plt.plot(dict_of_lists['round'], dict_of_lists[k])
        plt.title(k)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(fig_path), f'_{k}.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_model_to_wandb_artifact(state_dict, config, server_round, metrics):
    # Create a new wandb artifact
    artifact = wandb.Artifact(
        name=f"{config.dataset}_{config.model}", 
        type="model",
        description=f"Model checkpoint from round {server_round}"
    )

    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        # Save the model state along with metadata
        save_dict = {
            'model_state': state_dict,
            'server_round': server_round,
            'model_name': config.model,
            'metrics': metrics
        }
        torch.save(save_dict, tmp_file.name)
        
        # Add file to artifact
        artifact.add_file(tmp_file.name, name=f"model.pth")

    # Log artifact to wandb
    wandb.log_artifact(artifact)

    # Clean up the temporary file
    os.unlink(tmp_file.name)

    log(INFO, f"Model from round {server_round} saved to wandb artifact.")
