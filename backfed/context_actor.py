"""
Context actor for communication between malicious clients.
"""

import asyncio
import torch
import ray
from logging import INFO
from typing import Dict, Any
from backfed.utils import log

@ray.remote
class ContextActor:
    """
    Ray actor that serves as a communication tunnel for malicious clients.
    This allows malicious clients to share models, data, triggers, and other information
    when running in parallel mode.
    """
    def __init__(self):
        self.shared_resources = {}  # round_number -> resource_package
        self.ready_events = {}  # round_number -> Event
        log(INFO, "Context actor initialized in parallel mode")

    def update_resource(self, client_id: int, resource_package: Dict[str, Any], round_number: int):
        """
        Update the resource and notify waiters. This should be called after poison_warmup when trigger pattern (IBA) or trigger generator (A3FL) is updated.
        Args:
            client_id (int): The ID of the client updating the resource.
            resource_package (dict): New package to update the resource.
            round_number (int): The round number for which the resource is updated.
        Returns:
            dict: The updated resource package.
        """
        log(INFO, f"Client[{client_id}] updated resources for round {round_number}")
        self.shared_resources[round_number] = resource_package

        # Create and set event for this round if it doesn't exist
        if round_number not in self.ready_events:
            self.ready_events[round_number] = asyncio.Event()
        self.ready_events[round_number].set() # set the event to signal that the resource is updated

        # Clean up old rounds to prevent memory leaks
        self._cleanup_old_rounds(round_number)
        return resource_package

    async def wait_for_resource(self, round_number: int):
        """
        Wait for resources to be available for a specific round.
        Args:
            round_number (int): The round number for which to retrieve resources.
        Returns:
            dict: The resource package.
        """
        # If resource already exists, return it immediately
        if round_number in self.shared_resources:
            return self.shared_resources[round_number]
            
        # Create event for this round if it doesn't exist
        if round_number not in self.ready_events:
            self.ready_events[round_number] = asyncio.Event()
            
        # Wait for the resource to be available
        await self.ready_events[round_number].wait()
        
        # Return the resource (should exist now)
        return self.shared_resources[round_number]

    def _cleanup_old_rounds(self, current_round: int, keep_last: int = 10):
        """
        Clean up resources from old rounds to prevent memory leaks.
        Args:
            current_round (int): The current round number.
            keep_last (int): The number of most recent rounds to keep.
        """
        if len(self.shared_resources) <= keep_last:
            return
            
        # Get rounds to clean up
        round_keys = sorted(list(self.shared_resources.keys()))
        rounds_to_remove = round_keys[:-keep_last]
        
        # Clean up resources and events
        for round_num in rounds_to_remove:
            if round_num in self.shared_resources:
                del self.shared_resources[round_num]
            if round_num in self.ready_events:
                del self.ready_events[round_num]
                
        log(INFO, f"Cleaned up resources for rounds: {rounds_to_remove}")

    def get_memory_usage(self):
        """
        Get the current memory usage of the actor.
        Returns:
            dict: Memory usage statistics.
        """
        import psutil
        import os

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
            "num_resource_packages": len(self.shared_resources),
            "cuda_memory": cuda_memory
        }
