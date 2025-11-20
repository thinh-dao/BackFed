"""
ClientApp implementation for FL.
Separated from base_client.py to avoid circular imports.
"""

import time
import traceback
import psutil
import os
import gc
import torch
import torch.nn as nn
import copy

from backfed.clients import MaliciousClient
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import Subset, Dataset
from omegaconf import DictConfig
from backfed.utils import log, set_random_seed
from backfed.const import ModelUpdate, Metrics
from backfed.datasets import nonIID_Dataset
from logging import INFO

class ClientApp:
    """
    ClientApp is a wrapper around the client class, used mainly for Ray Actor initialization.
    It is used to load the client class based on the client_id.
    """
    def __init__(
        self,
        client_config: DictConfig,
        model: nn.Module,
        dataset: Optional[Dataset],
        dataset_partition: Optional[List[List[int]]],
        secret_dataset_indices: Optional[List[int]]
    ):
        """
        Initialize ClientApp with preloaded model and dataset for Ray Actor optimization.
        For nonIID datasets (FEMNIST, REDDIT, SENTIMENT140), dataset and dataset_partition will be None since clients will load their own data.
        Args:
            client_config: Default local training configuration for client
            model: Pre-initialized model to be copied for each client
            dataset: Dataset reference. 
            dataset_partition: List of indices for data partitioning.
            secret_dataset_indices: List of indices for malicious clients to do poison training (if any).
        """
        # Set random seed
        set_random_seed(seed=client_config.seed, deterministic=client_config.deterministic)
        
        self.base_model = model  # Store pre-initialized model
        self.dataset = dataset
        self.dataset_partition = dataset_partition
        self.secret_dataset_indices = secret_dataset_indices
        self.client_config = client_config
        self.client = None  # Will be set to BaseClient instance
        self.pool = None # ThreadPoolExecutor for timeout handling

    def _load_client(self, client_cls, client_id: int, **init_args):
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

        # Import here to avoid circular imports
        if self.dataset is None and self.dataset_partition is None:
            dataset = nonIID_Dataset(self.client_config.dataset, self.client_config, client_id)
        else:
            if issubclass(client_cls, MaliciousClient) and self.secret_dataset_indices is not None:
                dataset = Subset(self.dataset, self.secret_dataset_indices)
            else:
                dataset = Subset(self.dataset, self.dataset_partition[client_id])
        
        # Initialize client with deep copy of preloaded model
        return client_cls(
            client_id=client_id,
            dataset=dataset,
            model=copy.deepcopy(self.base_model),
            client_config=self.client_config,
            **init_args
        )

    def train(self, client_cls, client_id: int, init_args: Dict[str, Any], train_package: Dict[str, Any], timeout: Optional[float] = None) -> Tuple[int, ModelUpdate, Metrics]:
        try:
            # Clear memory before loading a new client
            if self.client is not None:
                self._cleanup_client()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            self.client = self._load_client(client_cls, client_id, **init_args)

            train_time_start = time.time()

            if timeout is not None:
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(max_workers=1)

                if self.pool is None:
                    raise ValueError("Pool is not initialized")

                future = self.pool.submit(self.client.train, train_package)
                results = future.result(timeout=timeout)
            else:
                results = self.client.train(train_package)
        except Exception as e:
            error_tb = traceback.format_exc()
            return 0.0, {
                "status": "failure",
                "error": str(e),
                "traceback": error_tb
            }

        assert len(results) == 3, "Training results must contain (num_examples, state_dict, training_metrics)"

        ram_usage = psutil.Process().memory_info().rss / (1024 ** 3)  # GB

        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_mem_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            peak_mem_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
        else:
            gpu_mem_allocated = 0.0
            peak_mem_allocated = 0.0

        log(
            INFO, 
            f"Client [{self.client.client_id}] ({self.client.client_type}) - "
            f"Training time: {train_time:.2f}s | RAM: {ram_usage:.2f}GB | "
            f"VRAM: {gpu_mem_allocated:.2f}GB | Peak VRAM: {peak_mem_allocated:.2f}GB"
        )

        return train_time, results

    def evaluate(self, test_package: Dict[str, Any], timeout: Optional[float] = None) -> Tuple[int, Metrics]:
        try:
            assert self.client is not None, "Only initialized client (after training) can be evaluated"

            eval_time_start = time.time()

            if timeout is not None:
                if self.pool is None:
                    self.pool = ThreadPoolExecutor(max_workers=1)

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

        return eval_time, results

    def _cleanup_client(self):
        """
        Clean up client resources to free memory. 
        """
        if self.client is None:
            return
            
        # Delete client reference and force garbage collection
        del self.client
        self.client = None
        
        # Release memory immediately
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
