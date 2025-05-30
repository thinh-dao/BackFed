
import math
import torch

from typing import List
from omegaconf import DictConfig
from backfed.poisons.base import Poison
from backfed.const import IMG_SIZE

# Default trigger settings per dataset
DEFAULT_TRIGGER_SETTINGS = {
    "MNIST": {"size": [1, 4], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 4},
    "CIFAR10": {"size": [1, 5], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 5},
    "CIFAR100": {"size": [1, 5], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 5},
    "TINYIMAGENET": {"size": [1, 10], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 10},
    "EMNIST_BYCLASS": {"size": [1, 4], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 4},
    "EMNIST_BALANCED": {"size": [1, 4], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 4},
    "EMNIST_DIGITS": {"size": [1, 4], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 4},
    "FEMNIST": {"size": [1, 4], "gap": [2, 2], "loc": [0, 0], "maximum_shares": 4}
}

class Distributed(Poison):
    """Each client has a unique trigger pattern."""
    
    def __init__(self, 
            params: DictConfig,  
            client_id: int = -1,
            trigger_size: List[int] = None,  # (height, width) of a local distributed trigger
            trigger_gap: List[int] = None,   # (gap_x, gap_y) Distance between distributed triggers
            trigger_loc: List[int] = None,   # (shift_x, shift_y) offset from top-left corner
            maximum_shares: int = None # Maximum trigger fragments
        ):
        super().__init__(params, client_id)
        
        # Initialize trigger parameters
        dataset_settings = DEFAULT_TRIGGER_SETTINGS.get(self.params['dataset'].upper(), {})
        self.trigger_size = trigger_size or dataset_settings["size"]
        self.trigger_gap = trigger_gap or dataset_settings["gap"]
        self.trigger_loc = trigger_loc or dataset_settings["loc"]
        self.maximum_shares = maximum_shares or dataset_settings["maximum_shares"]

        # Cache for all trigger positions
        self.trigger_positions = {}
        
        # Initialize trigger positions for all clients
        self.init_all_trigger_positions()
    
    def init_all_trigger_positions(self):
        """Pre-compute trigger positions for all malicious clients"""
        img_height, img_width, _ = IMG_SIZE[self.params['dataset'].upper()]
        malicious_clients = self.params.malicious_clients
        num_rows = int(math.sqrt(len(malicious_clients)))

        # Initialize server-side positions (client_id = -1)
        server_positions = {
            'start_x': [], 'end_x': [],
            'start_y': [], 'end_y': []
        }
        
        # Calculate positions for each malicious client
        for idx, client_id in enumerate(malicious_clients):
            if idx >= self.maximum_shares:
                idx = idx % self.maximum_shares
            
            # If either gap_x or gap_y is 0, we only slide the trigger fragment in one direction
            if self.trigger_gap[0] == 0 or self.trigger_gap[1] == 0:
                if self.trigger_gap[0] == 0:
                    row = 0
                    col = idx
                else:
                    col = 0
                    row = idx
            else:
                row, col = idx // num_rows, idx % num_rows
                
            start_x = self.trigger_loc[0] + (row * (self.trigger_gap[0] + self.trigger_size[0]))
            start_y = self.trigger_loc[1] + (col * (self.trigger_gap[1] + self.trigger_size[1]))
            end_x = start_x + self.trigger_size[0]
            end_y = start_y + self.trigger_size[1]
            
            assert start_x >= 0 and start_x < img_height, \
                f"Invalid trigger coordinate {start_x} for image height {img_height}"
            assert end_x >= 0 and end_x < img_height, \
                f"Invalid trigger coordinate {end_x} for image height {img_height}"
            assert start_y >= 0 and start_y < img_width, \
                f"Invalid trigger coordinate {start_y} for image width {img_width}"
            assert end_y >= 0 and end_y < img_width, \
                f"Invalid trigger coordinate {end_y} for image width {img_width}"
            
            # Store positions for individual client
            self.trigger_positions[client_id] = {
                'start_x': start_x, 'end_x': end_x,
                'start_y': start_y, 'end_y': end_y
            }
            
            # Append to server positions
            server_positions['start_x'].append(start_x)
            server_positions['end_x'].append(end_x)
            server_positions['start_y'].append(start_y)
            server_positions['end_y'].append(end_y)
        
        # Store server positions
        self.trigger_positions[-1] = server_positions
        
        # Pre-create server trigger mask (will be moved to correct device during inference)
        channels = 1  # Default for grayscale images
        if self.params['dataset'].upper() in ["CIFAR10", "CIFAR100", "TINYIMAGENET"]:
            channels = 3  # RGB images
            
        self.server_trigger_mask = torch.zeros((channels, img_height, img_width), device=self.device)
        for start_x, end_x, start_y, end_y in zip(
            server_positions['start_x'], server_positions['end_x'],
            server_positions['start_y'], server_positions['end_y']):
            self.server_trigger_mask[:, start_x:end_x, start_y:end_y] = 1.0

    def poison_inputs(self, inputs):
        """Apply trigger pattern to inputs"""
        poison_inputs = inputs.clone()
        positions = self.trigger_positions[self.client_id]
        
        if self.client_id != -1:
            # Client-side: Single assignment for one trigger
            poison_inputs[:, :, 
                         positions['start_x']:positions['end_x'],
                         positions['start_y']:positions['end_y']] = 1.0
        else:
            # Ensure server trigger mask is on the same device as inputs
            if self.server_trigger_mask.device != inputs.device:
                self.server_trigger_mask = self.server_trigger_mask.to(inputs.device)
                
            # Apply the 3D mask to each sample in the batch
            mask = self.server_trigger_mask.unsqueeze(0).expand_as(poison_inputs)
            poison_inputs = torch.where(
                mask == 1,
                torch.ones_like(poison_inputs),
                poison_inputs
            )
        return poison_inputs
        
class Centralized(Distributed):
    """Each client has similar trigger pattern - the aggregated trigger pattern."""
    
    def poison_inputs(self, inputs):
        """Apply trigger pattern to inputs"""
        poison_inputs = inputs.clone()
        positions = self.trigger_positions[-1] # Server-side positions
        
        # Server-side: Use server trigger mask if available, otherwise create it
        if self.server_trigger_mask is None:
            self.server_trigger_mask = torch.zeros_like(inputs)
            for start_x, end_x, start_y, end_y in zip(
                positions['start_x'], positions['end_x'],
                positions['start_y'], positions['end_y']):
                self.server_trigger_mask[:, :, start_x:end_x, start_y:end_y] = 1.0
        
        if self.server_trigger_mask.device != inputs.device:
            self.server_trigger_mask = self.server_trigger_mask.to(inputs.device)
            
        # Apply all triggers using the server trigger mask
        poison_inputs = torch.where(
            self.server_trigger_mask == 1,
            torch.ones_like(poison_inputs),
            poison_inputs
        )

        return poison_inputs
