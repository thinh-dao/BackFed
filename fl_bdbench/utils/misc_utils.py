"""
Miscellaneous utilities for FL.
"""
import functools
import asyncio
import inspect
import torch

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Dict

# Method 1: Convert async to sync function
def async_to_sync(func: Callable) -> Callable:
    """Transforms an async function to return a sync function."""
    def sync_func(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    
    functools.update_wrapper(sync_func, func)
    return sync_func

# Method 2: Convert sync to async function
def sync_to_async(func: Callable) -> Callable:
    pool = ThreadPoolExecutor()
    
    async def async_func(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, func, *args, **kwargs)
    
    functools.update_wrapper(async_func, func)
    return async_func

def with_timeout(func: Callable, timeout: Optional[float] = None) -> Callable:
    """
    Transform any function (sync or async) to include timeout functionality.
    
    Args:
        func: The function to transform (can be sync or async)
        timeout: Timeout in seconds. If None, no timeout is applied.
    Returns:
        A new async function with timeout functionality
    """
    # Convert to async if it's a sync function
    async_func = func if inspect.iscoroutinefunction(func) else sync_to_async(func)
    
    async def timeout_wrapper(*args, **kwargs) -> Dict:
        try:
            async with asyncio.timeout(timeout) if timeout else asyncio.nullcontext():
                return await async_func(*args, **kwargs)
        except asyncio.TimeoutError:
            # Assuming first arg is self for class methods
            worker_id = getattr(args[0], 'worker_id', None) if args else None
            return {
                "worker_id": worker_id,
                "task_id": kwargs.get("task_id", args[1] if len(args) > 1 else None),
                "status": "timeout",
                "mode": "async",
                "timeout": timeout
            }
    return timeout_wrapper

def test(net, test_loader, device, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
    """Validate the model performance on the test set."""
    net.eval()
    net.to(device)
    correct, loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if normalization:
                inputs = normalization(inputs)
                
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(inputs)
            loss += loss_fn(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total_samples += len(inputs)
    accuracy = correct / total_samples
    loss = loss / len(test_loader)
    return loss, accuracy

def format_time_hms(seconds: float) -> str:
    """
    Format time in seconds to hours, minutes, seconds string.
    
    Args:
        seconds: Time in seconds
    Returns:
        Formatted string in the format "XXh YYm ZZs"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}h {minutes:02d}m {seconds:02.0f}s"
