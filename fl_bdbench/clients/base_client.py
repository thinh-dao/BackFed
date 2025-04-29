"""
Base client implementation for FL.
"""

import random
import torch
import torch.nn as nn
import time

from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import DataLoader, Subset, Dataset
from omegaconf import DictConfig
from fl_bdbench.utils import set_random_seed, with_timeout, log
from fl_bdbench.const import StateDict, Metrics
from hydra.utils import instantiate
from logging import INFO

class BaseClient:
    """
    Base class for all FL clients.
    Handles data partitioning, model setup, optimizer, and training logic.
    """
    def __init__(self, client_id: int, dataset: Dataset, dataset_indices: List[List[int]], 
                 model: nn.Module, client_config: DictConfig, client_type: str = "base", verbose: bool = True, **kwargs):
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

        # Set up model, dataloader, optimizer, criterion
        self.model = model.to(self.device)
        self._set_dataloader(dataset, self.client_indices)
        self._set_optimizer()
        self._set_criterion()

        # Resource metrics
        self.training_time = 0.0
        self.current_memory = 0.0
        self.max_memory = 0.0

        set_random_seed(seed=self.client_config.seed, deterministic=True)

        # Wrap the train method with timeout if a timeout is specified
        if self.client_config.timeout is not None:
            self.train = with_timeout(self.train, self.client_config.timeout)

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
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["train_batch_size"], shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.client_config["val_batch_size"], shuffle=True, pin_memory=True)
        else:
            self.train_dataset = Subset(dataset, indices)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.client_config["train_batch_size"], shuffle=True, pin_memory=True)

    def get_model_parameters(self) -> StateDict:
        """
        Get the global model parameters.
        """
        return {name: param.cpu() for name, param in self.model.state_dict().items()}
    
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        """
        Train the model for a number of epochs.
        Args:
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        Returns:
            (num_examples, state_dict, metrics). The return state_dict must be on cpu.
        """
        raise NotImplementedError("Train method must be implemented by subclasses")

    def evaluate(self, test_package: Dict[str, Any]) -> Tuple[int, Metrics]:
        """
        Evaluate the model on test data.
        Args:
            test_package: Data package received from server to evaluate the model (e.g., global model weights, learning rate, etc.)
        Returns:
            (num_examples, metrics).
        """
        raise NotImplementedError("Evaluate method must be implemented by subclasses")

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

    def train(self, client_cls: BaseClient, client_id: int, init_args: Optional[Dict[str, Any]] = None, train_package: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the client with preloaded model.
        Args:
            client_cls: Client class to be loaded
            client_id: Unique identifier for the client
            init_args: Additional keyword arguments for client initialization
            train_package: Data package received from server to train the model (e.g., global model weights, learning rate, etc.)
        Returns:
            Dictionary containing training metrics
        """
        time_start = time.time()
        self.client = self._load_client(client_cls, client_id, **init_args)
        time_end = time.time()
        time_load = time_end - time_start
        log(INFO, f"Client loading time: {time_load:.2f} seconds")
       
        time_start = time.time()
        results = self.client.train(train_package)
        time_end = time.time()
        time_train = time_end - time_start
        log(INFO, f"Client training time: {time_train:.2f} seconds")
        return results

    def evaluate(self, test_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the client with preloaded model.
        Args:
            test_package: Data package received from server to evaluate the model (e.g., global model weights, learning rate, etc.)
        Returns:
            Dictionary containing evaluation metrics
        """
        assert self.client is not None, "Only initialized client (after training) can be evaluated"
        return self.client.evaluate(test_package)

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
