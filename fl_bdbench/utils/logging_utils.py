"""
Logging utilities for FL.
"""

import os
import logging
import csv
import hydra
import omegaconf
import ray
import torch
import wandb
import tempfile
import matplotlib.pyplot as plt
import re

from dotenv import load_dotenv
from omegaconf import OmegaConf
from logging import INFO
from rich.logging import RichHandler
from rich.console import Console

# Create a global console for rich output
rich_console = Console(stderr=True, width=None)  

# Configure root logger once at module import time
def _setup_logging():
    """Set up logging configuration once"""
    root_logger = logging.getLogger('fl_bdbench')
    root_logger.setLevel(logging.INFO)
    
    # Check if rich handler is already configured
    has_rich_handler = any(isinstance(h, RichHandler) for h in root_logger.handlers)
    
    if not has_rich_handler:
        # Remove any existing handlers to avoid duplicates
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            
        # Add Rich handler to root logger
        rich_handler = RichHandler(
            console=rich_console,
            show_time=False,
            show_path=False,
            show_level=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_suppress=[hydra, omegaconf, ray, torch]
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(rich_handler)
    
    return root_logger

# Initialize the logger once
logger = _setup_logging()

def log(level, message, *args, **kwargs):
    """Log a message at the specified level."""
    if isinstance(message, dict):
        formatted_dict = "\n".join(f"    {k}: {v}" for k, v in message.items())
        logger.log(level, formatted_dict, *args, **kwargs)
    else:
        logger.log(level, message, *args, **kwargs)

def log_runtime(start_time, end_time, phase="Total"):
    """Log runtime information for a specific phase."""
    runtime = end_time - start_time
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"
        
    logger.info(f"[Runtime] {phase}: {time_str}")
    return runtime

def get_console():
    """Get a properly configured console instance that works with Hydra's rich logging.
    
    Since Hydra is already configured with rich logging in the YAML file,
    we should reuse the existing console rather than creating a new one.
    
    Returns:
        Console: A configured rich console instance
    """
    return rich_console

def create_rich_console():
    """Create a Rich console for logging.
    
    This function is used by Hydra's configuration system to create a properly
    instantiated Console object for the RichHandler.
    
    Returns:
        Console: A configured Rich console instance
    """
    from rich.console import Console
    return Console(stderr=True)

class CSVLogger():
    def __init__(self, fieldnames, resume=False, filename='log.csv'):
        self.filename = filename
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Track current step/round for logging
        self.current_step = 0
        # Buffer for accumulating metrics at current step
        self.current_metrics = {}
        
        resume = os.path.exists(self.filename) and resume == True
        if resume:
            self.csv_file = open(filename, 'a')
        else:
            self.csv_file = open(filename, 'w')
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not resume:
            self.writer.writeheader()
            self.csv_file.flush()
        self.first_step = True

    def log(self, metrics, step=None):
        """
        Log metrics in wandb-like style.
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number. If None, uses internal counter
        """
        if step is not None and step != self.current_step and self.current_metrics:
            # If we're moving to a new step, write the accumulated metrics
            self._write_current_metrics()
            self.current_step = step
            self.current_metrics = {}
        
        # Update current metrics with new values
        if self.first_step:
            self.first_step = False
            self.current_step = step
        self.current_metrics.update(metrics)
        
        if step is None:
            # If no step provided, write immediately
            self._write_current_metrics()
            self.current_step += 1
            self.current_metrics = {}

    def _write_current_metrics(self):
        """Write accumulated metrics to CSV file"""
        # Ensure all fieldnames have a value (use None if not provided)
        row = {field: None for field in self.fieldnames}
        row.update(self.current_metrics)
        
        # Add the step/round if it's part of fieldnames
        if 'round' in self.fieldnames:
            row['round'] = self.current_step
        
        self.writer.writerow(row)
        self.csv_file.flush()

    def flush(self):
        """Force write of current metrics"""
        if self.current_metrics:
            self._write_current_metrics()
            self.current_metrics = {}

    def finish(self):
        self.flush()
        self.csv_file.close()

def init_csv_logger(config, resume=False, detection=False):
    if config.partitioner.lower() == "dirichlet":
        partitoner = f"dirichlet({config.alpha})"
    else:
        partitoner = "uniform"

    aggregator = config.aggregator

    if config.no_attack:
        attack_name = "noattack"
    else:
        attack_name = f"{config.atk_config.model_poison_method}({config.atk_config.data_poison_method})"

    name = f"{attack_name}_{aggregator.lower()}_{config.dataset.lower()}_{partitoner}_{config.atk_config.selection_scheme}_{config.atk_config.poison_frequency}"
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
    if config.federated_evaluation:
        field_names.extend(["val_clean_loss", "val_clean_acc", "val_backdoor_loss", "val_backdoor_acc"])
    if detection:
        field_names.extend(["precision", "recall", "f1_score", "fpr", "fpr_clean"])

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

    config.wandb.name = f"{attack_name}_{aggregator.lower()}_{config.dataset.lower()}_{partitoner}_{config.atk_config.selection_scheme}_{config.atk_config.poison_frequency}"
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
