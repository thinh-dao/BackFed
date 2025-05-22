"""
Base client implementation for FL.
"""

import random
import torch
import torch.nn as nn
import time
import traceback
import psutil
import os

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import DataLoader, Subset, Dataset
from omegaconf import DictConfig
from backfed.utils import set_random_seed, log
from backfed.const import StateDict, Metrics
from hydra.utils import instantiate
from logging import INFO

class BaseClient:
    """
    Base class for all FL clients.
    Handles data partitioning, model setup, optimizer, and training logic.
    """
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        dataset_indices: List[List[int]],
        model: nn.Module,
        client_config: DictConfig,
        client_type: str = "base",
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the client.
        Args:
            client_id: Unique identifier
            dataset: The whole training dataset
            dataset_indices: Data indices for all clients (list of lists)
            model: Training model
            client_config: Dictionary containing training configuration
            client_type: String for client type identification
        """
        self.client_id = client_id
        self.client_config = client_config
        self.client_type = client_type
        self.client_indices = dataset_indices[client_id]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Set random seed
        set_random_seed(seed=self.client_config.seed, deterministic=self.client_config.deterministic)

        # Set up model, dataloader, optimizer, criterion
        self.model = model.to(self.device)
        self._set_dataloader(dataset, self.client_indices)
        self._set_optimizer()
        self._set_criterion()

        # Resource metrics
        self.training_time = 0.0
        self.current_memory = 0.0
        self.max_memory = 0.0

        # Training and evaluation metrics
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.eval_loss = 0.0
        self.eval_accuracy = 0.0

        # Reset peak memory stats to track memory usage of this client
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _set_optimizer(self):
        """
        Set up the optimizer for the client. Uses the optimizer specified in the config.
        """
        self.optimizer = instantiate(self.client_config.optimizer, params=self.model.parameters())

    def _set_criterion(self):
        """
        Set up the loss criterion for the client. Defaults to CrossEntropyLoss.
        """
        self.criterion = nn.CrossEntropyLoss()

    def _set_dataloader(self, dataset, indices):
        """
        Set up train and validation data loaders for the client.
        """
        if self.client_config.val_split > 0.0:
            num_val = int(len(indices) * self.client_config.val_split)
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)
            val_indices = shuffled_indices[:num_val]
            train_indices = shuffled_indices[num_val:]

            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)
        else:
            self.train_dataset = Subset(dataset, indices)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["batch_size"], shuffle=True, pin_memory=False)

    def _check_required_keys(self, train_package: Dict[str, Any], required_keys: List[str] = ["global_model_params", "server_round"]):
        """
        Check if the required keys are present in the train_package.
        """
        for key in required_keys:
            assert key in train_package, f"{key} not found in train_package for {self.client_type} client"

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        """
        Train the model for a number of epochs.

        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)

        Returns:
            num_examples (int): number of examples in the training dataset
            state_dict (StateDict): updated model parameters
            training_metrics (Dict[str, float]): training metrics
        """
        raise NotImplementedError("Train method must be implemented by subclasses")

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model on test data.
        Args:
            test_package: Data package received from server to evaluate the model (e.g., global model weights, learning rate, etc.)
        Returns:
            num_examples (int): number of examples in the test dataset
            evaluation_metrics (Dict[str, float]): evaluation metrics
        """
        raise NotImplementedError("Evaluate method must be implemented by subclasses")

    # An utility function to calculate the L2 distance between client model parameters and global parameters
    def model_dist(self, global_params_tensor: torch.Tensor, client_model=None, gradient_calc=False):
        """Calculate the L2 distance between client model parameters and global parameters"""
        if client_model is None:
            client_model = self.model

        client_params_tensor = torch.cat([param.view(-1) for param in client_model.parameters()]).to(self.device)
        global_params_tensor = global_params_tensor.to(self.device)
        if gradient_calc:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2).item()
        else:
            return torch.linalg.norm(client_params_tensor - global_params_tensor, ord=2)

    def get_model_parameters(self) -> StateDict:
        """
        Move the global model parameters to cpu.
        """
        return {name: param.cpu() for name, param in self.model.state_dict().items()}

    def get_resource_metrics(self):
        """
        Get resource usage metrics.
        Returns:
            Dictionary containing resource metrics
        """
        # Get current GPU memory usage if available
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            self.max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        return {
            "last_training_time": self.training_time,
            "current_gpu_memory": self.current_memory,
            "max_gpu_memory": self.max_memory
        }

    def get_client_info(self):
        """
        Get client information.
        Returns:
            Dictionary containing client information
        """
        return {
            "client_id": self.client_id,
            "client_type": self.client_type,
            "device": str(self.device),
            "dataset_size": len(self.train_dataset)
        }

    def get_client_type(self):
        """
        Get client type.
        Returns:
            String for client type identification
        """
        return self.client_type

class ClientApp:
    """
    ClientApp is a wrapper around the client class, used mainly for Ray Actor initialization.
    It is used to load the client class based on the client_id.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        dataset_indices: List[int],
        client_config: DictConfig
    ) -> None:
        """
        Initialize ClientApp with preloaded model and dataset for Ray Actor optimization.
        Args:
            model: Pre-initialized model to be copied for each client
            dataset: Dataset reference
            dataset_indices: List of indices for data partitioning
            client_config: Default local training configuration for client
        """
        self.base_model = model  # Store pre-initialized model
        self.dataset = dataset
        self.dataset_indices = dataset_indices
        self.client_config = client_config
        self.client : Optional[BaseClient] = None

        if self.client_config.timeout is not None:
            self.pool = ThreadPoolExecutor(max_workers=1) # Only one worker for timeout
        else:
            self.pool = None

    def _load_client(self, client_cls, client_id: int, **init_args) -> BaseClient:
        """
        Load appropriate client based on client_id, using the preloaded model.
        Args:
            client_cls: Client class to be loaded
            client_id: Unique identifier for the client
            **init_args: Additional keyword arguments for client initialization
        Returns:
            Loaded client instance
        """
        if client_cls is None:
            raise ValueError(f"Client class must be provided")

        # Initialize client with deep copy of preloaded model
        return client_cls(
            client_id=client_id,
            dataset=self.dataset,
            dataset_indices=self.dataset_indices,
            model=self.base_model,
            client_config=self.client_config,
            **init_args
        )

    def train(self, client_cls: BaseClient, client_id: int, init_args: Dict[str, Any], train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        try:
            # Clear memory before loading a new client
            if self.client is not None:
                self._cleanup_client()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.client = self._load_client(client_cls, client_id, **init_args)

            train_time_start = time.time()
            timeout = self.client.client_config.timeout
            if timeout is not None:
                if self.pool is None:
                    raise ValueError("Pool is not initialized")

                future = self.pool.submit(self.client.train, train_package)
                results = future.result(timeout=timeout)
            else:
                results = self.client.train(train_package)
        except Exception as e:
            error_tb = traceback.format_exc()
            return {
                "status": "failure",
                "error": str(e),
                "traceback": error_tb
            }

        assert len(results) == 3, "Training results must contain (num_examples, state_dict, training_metrics)"

        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        log(INFO, f"Client [{self.client.client_id}] ({self.client.client_type}) - Training time: {train_time:.2f} seconds")

        return results

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        try:
            assert self.client is not None, "Only initialized client (after training) can be evaluated"

            eval_time_start = time.time()
            timeout = self.client.client_config.timeout
            if timeout is not None:
                if self.pool is None:
                    raise ValueError("Pool is not initialized")

                future = self.pool.submit(self.client.evaluate, test_package)
                results = future.result(timeout=timeout)
            else:
                results = self.client.evaluate(test_package)
        except Exception as e:
            error_tb = traceback.format_exc()
            return {
                "status": "failure",
                "error": str(e),
                "traceback": error_tb
            }

        eval_time_end = time.time()
        eval_time = eval_time_end - eval_time_start
        log(INFO, f"Client [{self.client.client_id}] ({self.client.client_type}) - Evaluation time: {eval_time:.2f} seconds")

        return results

    def execute(self, client_cls: BaseClient, client_id: int, init_args: Dict[str, Any], exec_package: Dict[str, Any]) -> Any:
        """
        Execute the client with preloaded model.
        """
        self.client = self._load_client(client_cls, client_id, **init_args)
        return self.client.execute(exec_package)

    def _cleanup_client(self):
        """
        Clean up client resources to free memory.
        """
        if self.client is None:
            return

        # Free GPU memory for model tensors
        if hasattr(self.client, 'model') and self.client.model is not None:
            for param in self.client.model.parameters():
                if param.is_cuda:
                    param.data = param.data.cpu()
                    if param.grad is not None:
                        param.grad.data = param.grad.data.cpu()

            # Clear model references
            self.client.model = None

        # Clear dataloader references
        if hasattr(self.client, 'train_loader'):
            self.client.train_loader = None
        if hasattr(self.client, 'val_loader'):
            self.client.val_loader = None

        # Clear optimizer state
        if hasattr(self.client, 'optimizer'):
            self.client.optimizer = None

        # Set client to None
        self.client = None

    def get_memory_usage(self):
        """
        Get the current memory usage of the actor.
        Returns:
            dict: Memory usage statistics.
        """

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Get CUDA memory if available
        cuda_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cuda_memory[f"cuda:{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / (1024 ** 2),  # MB
                    "cached": torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                }

        return {
            "rss": memory_info.rss / (1024 ** 2),  # MB
            "vms": memory_info.vms / (1024 ** 2),  # MB
            "shared": getattr(memory_info, "shared", 0) / (1024 ** 2),  # MB
            "cuda_memory": cuda_memory
        }

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the client if not found in ClientApp.
        Args:
            name: Attribute name
        Returns:
            Attribute value
        """
        if self.client is not None:
            return getattr(self.client, name)
        raise AttributeError(f"'{self.__class__.__name__}' object and its client have no attribute '{name}'")
