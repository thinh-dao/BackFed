"""
Context actor for communication between malicious clients.
"""

import asyncio

from logging import INFO
from typing import Dict, Any
from fl_bdbench.utils import log

class ContextActor:
    """
    Ray actor that serves as a communication tunnel for malicious clients.
    This allows malicious clients to share models, data, triggers, and other information
    when running in parallel mode.
    """
    def __init__(self):
        self.shared_resources = {}  # round_number -> resource_package
        self.ready_event = asyncio.Event()
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
        # Clean up old rounds to prevent memory leaks
        self._cleanup_old_rounds(round_number)
        # Signal that the resource is updated
        self.ready_event.set()
        return resource_package

    async def wait_for_resource(self, round_number: int):
        """
        Wait for resources to be available.
        Args:
            round_number (int): The round number for which to retrieve resources.
        Returns:
            dict: The resource package.
        """
        if round_number not in self.shared_resources:
            await self.ready_event.wait()
        return self.shared_resources[round_number]

    def _cleanup_old_rounds(self, keep_last: int = 10):
        """
        Clean up resources from old rounds to prevent memory leaks.
        Args:
            keep_last (int): The number of most recent rounds to keep.
        """
        if len(self.shared_resources) <= keep_last:
            return
        else:
            round_keys = sorted(list(self.shared_resources.keys()))
            for round_num in round_keys[:-keep_last]:
                del self.shared_resources[round_num]
