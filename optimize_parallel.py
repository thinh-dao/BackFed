"""
Script to optimize the FL framework for better performance in parallel mode.
"""

import os
import argparse
import torch
import ray
from typing import Dict, List, Any, Tuple, Optional
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_ray_config():
    """
    Optimize Ray configuration for better performance.
    """
    # Calculate optimal object store size based on available memory
    import psutil
    total_memory = psutil.virtual_memory().total
    # Use 60% of total memory for object store
    object_store_memory = int(total_memory * 0.6)
    
    # Initialize Ray with optimized configuration
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=object_store_memory,
        _memory=int(total_memory * 0.2),  # 20% for Ray heap memory
        num_cpus=None,  # Use all available CPUs
    )
    
    logger.info(f"Initialized Ray with object_store_memory={object_store_memory / (1024**3):.2f} GB")

def optimize_config(config: DictConfig) -> DictConfig:
    """
    Optimize configuration for better performance.
    
    Args:
        config: Original configuration
        
    Returns:
        Optimized configuration
    """
    # Create a copy of the config to avoid modifying the original
    optimized_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # Optimize resource allocation
    # Calculate optimal CPU and GPU allocation based on system resources
    import multiprocessing
    total_cpus = multiprocessing.cpu_count()
    total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Allocate resources more efficiently
    # Use fewer CPUs per client to allow more parallel clients
    optimized_config.num_cpus = max(1, total_cpus // 8)
    
    # For GPUs, allocate smaller fractions to allow more parallel clients
    if total_gpus > 0:
        optimized_config.num_gpus = max(0.1, total_gpus / 8)
    
    logger.info(f"Optimized resource allocation: {optimized_config.num_cpus} CPUs, {optimized_config.num_gpus} GPUs per client")
    
    # Optimize batch size for better GPU utilization
    if hasattr(optimized_config, "client_config") and hasattr(optimized_config.client_config, "train_batch_size"):
        # Increase batch size for better GPU utilization
        optimized_config.client_config.train_batch_size = min(256, optimized_config.client_config.train_batch_size * 2)
        logger.info(f"Optimized batch size: {optimized_config.client_config.train_batch_size}")
    
    # Enable mixed precision training for better performance
    if hasattr(optimized_config, "client_config"):
        optimized_config.client_config.mixed_precision = True
        logger.info("Enabled mixed precision training")
    
    return optimized_config

def patch_ray_functions():
    """
    Patch Ray functions for better performance.
    """
    # Store original functions
    original_ray_put = ray.put
    
    # Define optimized functions
    def optimized_ray_put(obj):
        """Optimized version of ray.put that avoids unnecessary copies."""
        if isinstance(obj, torch.Tensor):
            # Move tensor to CPU before putting in object store
            obj = obj.cpu()
        return original_ray_put(obj)
    
    # Apply patches
    ray.put = optimized_ray_put
    
    logger.info("Patched Ray functions for better performance")

def patch_fl_framework():
    """
    Patch FL framework for better performance.
    """
    try:
        # Import necessary modules
        from fl_bdbench.servers.base_server import FLTrainer
        
        # Store original methods
        original_parallel_train = FLTrainer._parallel_train
        
        # Define optimized methods
        def optimized_parallel_train(self, clients_mapping):
            """Optimized version of _parallel_train that uses batching for better performance."""
            idle_workers = list(range(self.num_workers))
            futures = []
            job_map = {}
            client_packages = {}
            
            # Prepare all clients and their corresponding packages
            all_clients = []
            for client_cls, clients in clients_mapping.items():
                init_args, train_package = self.server.train_package(client_cls)
                
                # Use a single reference for all clients of the same class
                init_args_ref = ray.put(init_args)
                train_package_ref = ray.put(train_package)
                
                for client_id in clients:
                    all_clients.append((client_cls, client_id, init_args_ref, train_package_ref))
            
            # Process clients in batches for better efficiency
            batch_size = max(1, len(idle_workers) // 2)
            for i in range(0, len(all_clients), batch_size):
                batch = all_clients[i:i+batch_size]
                batch_futures = []
                
                # Launch batch of tasks
                for j, (client_cls, client_id, init_args, train_package) in enumerate(batch):
                    if j >= len(idle_workers):
                        break
                        
                    worker_id = idle_workers[j]
                    future = self.workers[worker_id].train.remote(
                        client_cls=client_cls,
                        client_id=client_id,
                        init_args=init_args,
                        train_package=train_package
                    )
                    job_map[future] = (client_id, worker_id)
                    batch_futures.append(future)
                
                # Wait for batch to complete
                ready_futures, _ = ray.wait(batch_futures, num_returns=len(batch_futures))
                
                # Process results
                for future in ready_futures:
                    client_id, worker_id = job_map[future]
                    package = ray.get(future)
                    idle_workers.append(worker_id)
                    client_packages[client_id] = package
            
            return client_packages
        
        # Apply patches
        FLTrainer._parallel_train = optimized_parallel_train
        
        logger.info("Patched FL framework for better performance")
        
    except ImportError as e:
        logger.warning(f"Failed to patch FL framework: {e}")

def run_optimized_experiment(config_path=None):
    """
    Run FL experiment with optimized configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    # Optimize Ray configuration
    optimize_ray_config()
    
    # Patch Ray functions
    patch_ray_functions()
    
    # Patch FL framework
    patch_fl_framework()
    
    try:
        # Load configuration
        if config_path:
            config = OmegaConf.load(config_path)
        else:
            # Use default config
            from hydra.core.config_store import ConfigStore
            cs = ConfigStore.instance()
            config = cs.load("defaults")
        
        # Force parallel mode
        config.mode = "parallel"
        
        # Optimize configuration
        optimized_config = optimize_config(config)
        
        logger.info(f"Starting optimized experiment")
        
        # Initialize server
        aggregator = optimized_config["aggregator"]
        from fl_bdbench.servers.base_server import BaseServer
        server = instantiate(optimized_config.aggregator_config[aggregator], server_config=optimized_config, _recursive_=False)
        
        # Run experiment
        server.run_experiment()
        
    finally:
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized FL experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    run_optimized_experiment(args.config)
