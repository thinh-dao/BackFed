"""
Logging utilities for FL.
"""

import os
import logging
import csv
import hydra
import omegaconf
import ray

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


class FLLogger:
    """Logger for FL in distributed and serial modes."""
    _instances = {}
    _console = Console(stderr=True)
    
    @classmethod
    def get_logger(cls, name="fl_logger", log_level=logging.INFO):            
        if name in cls._instances:
            return cls._instances[name]
        
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Rich console handler for colored output
        console_handler = RichHandler(
            console=cls._console,
            show_time=False,
            show_path=False,
            show_level=True,  # Let Rich handle the level
            markup=True,
            rich_tracebacks=True,
            tracebacks_suppress=[
                hydra,
                omegaconf,
                ray
            ]
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        # Plain file handler for clean output
        try:
            from hydra.core.hydra_config import HydraConfig
            if HydraConfig.initialized():
                file_handler = logging.FileHandler(os.path.join(HydraConfig.get().runtime.output_dir, "main.log"))
                file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                logger.addHandler(file_handler)
        except ImportError:
            pass
        
        cls._instances[name] = logger
        return logger
    
    @staticmethod
    def log(level, message, logger_name="fl_logger", *args, **kwargs):
        """Log a message at the specified level."""
        logger = FLLogger.get_logger(logger_name)
        if isinstance(message, dict):
            formatted_dict = "\n" + "\n".join(f"    {k}: {v}" for k, v in message.items())
            logger.log(level, formatted_dict, *args, **kwargs)
        else:
            logger.log(level, message, *args, **kwargs)

    @classmethod
    def get_console(cls):
        """Get the shared console instance."""
        return cls._console

def log(*args, **kwargs):
    return FLLogger.log(*args, **kwargs)

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

def init_csv_logger(config, attack_name, strategy_name, resume=False, validation=False):
    if config.partitioner.lower() == "dirichlet":
        partitoner = f"dirichlet({config.alpha})"
    else:
        partitoner = "uniform"

    name = f"{attack_name}_{strategy_name.lower()}_{config.dataset.lower()}_{partitoner}_{config.atk_config.selection_scheme}_{config.atk_config.poison_scheme}"
    if config.name_tag:
        name = f"{name}_{config.name_tag}"

    dir_path = f"{attack_name}_{strategy_name.lower()}_{config.dataset.lower()}"
    if config.dir_tag:
        dir_path = os.path.join(str(config.dir_tag), dir_path)

    # Check if the run already exists and increment the version if it does
    dir_path = os.path.join("csv_results", dir_path)
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

class ColorFormatter(logging.Formatter):
    """Formatter that adds color for console output only"""
    def format(self, record):
        # Only add color markup for console
        if record.levelno >= logging.ERROR:
            record.levelname = "[red]ERROR[/red]"
        elif record.levelno >= logging.WARNING:
            record.levelname = "[yellow]WARNING[/yellow]"
        else:
            record.levelname = "[green]INFO[/green]"
        return super().format(record)
