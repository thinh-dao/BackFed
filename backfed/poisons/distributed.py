
import torch
import os

from torchvision.utils import save_image
from typing import List, Tuple
from omegaconf import DictConfig
from backfed.poisons.base import Poison
from backfed.const import IMG_SIZE

# MNIST: 4 shares, each pattern of size 4
MNIST_PATTERN = [
    [(1, i) for i in range(1, 5)],
    [(3, i) for i in range(1, 5)],
    [(5, i) for i in range(1, 5)],
    [(7, i) for i in range(1, 5)],
]

# CIFAR: 5 shares, each pattern of size 5
CIFAR_PATTERN = [
    [(1, i) for i in range(1, 6)],
    [(3, i) for i in range(1, 6)],
    [(5, i) for i in range(1, 6)],
    [(7, i) for i in range(1, 6)],
    [(9, i) for i in range(1, 6)],
]

# TinyImageNet: 5 shares, each pattern of size 2x10
TINYIMAGENET_PATTERN = [
    [(i, j) for j in range(1, 11) for i in range(1, 3)],
    [(i, j) for j in range(1, 11) for i in range(4, 6)],
    [(i, j) for j in range(1, 11) for i in range(7, 9)],
    [(i, j) for j in range(1, 11) for i in range(10, 12)],
    [(i, j) for j in range(1, 11) for i in range(13, 15)]
]

# Default trigger patterns per dataset (referencing base patterns)
DEFAULT_TRIGGER_PATTERNS = {
    "MNIST": MNIST_PATTERN,
    "EMNIST_BYCLASS": MNIST_PATTERN,
    "EMNIST_BALANCED": MNIST_PATTERN,
    "EMNIST_DIGITS": MNIST_PATTERN,
    "FEMNIST": MNIST_PATTERN,
    "CIFAR10": CIFAR_PATTERN,
    "CIFAR100": CIFAR_PATTERN,
    "TINYIMAGENET": TINYIMAGENET_PATTERN,
}

class Distributed(Poison):
    """Each client has a unique trigger pattern defined by explicit pixel coordinates."""
    
    def __init__(self, 
            params: DictConfig,  
            client_id: int = -1,
            trigger_patterns: List[List[Tuple[int, int]]] = None,  # Custom patterns for each share
            save_poisoned_images: bool = False,  # Flag to save images
            save_dir: str = "backfed/poisons/saved/distributed"  # Directory to save images
        ):
        super().__init__(params, client_id)
        
        # Get dataset-specific trigger patterns
        dataset = self.params['dataset'].upper()
        self.trigger_patterns = trigger_patterns or DEFAULT_TRIGGER_PATTERNS.get(dataset, DEFAULT_TRIGGER_PATTERNS["MNIST"])
        
        # Validate patterns
        img_height, img_width, _ = IMG_SIZE[dataset]
        for share_idx, pattern in enumerate(self.trigger_patterns):
            for x, y in pattern:
                assert 0 <= x < img_height, \
                    f"Share {share_idx}: Invalid x coordinate {x} for image height {img_height}"
                assert 0 <= y < img_width, \
                    f"Share {share_idx}: Invalid y coordinate {y} for image width {img_width}"

        # Map client IDs to their trigger patterns
        self.client_trigger_map = {}
        self.init_client_trigger_map()
        
        # Pre-create trigger masks
        self.init_trigger_masks()
        
        # Image saving settings
        self.save_poisoned_images = save_poisoned_images
        self.save_dir = save_dir
        self.saved_image_count = 0  # Track how many images saved
        if self.save_poisoned_images:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def init_client_trigger_map(self):
        """Map each malicious client to their trigger pattern (share)."""
        malicious_clients = self.params.malicious_clients
        num_shares = len(self.trigger_patterns)
        
        for idx, client_id in enumerate(malicious_clients):
            # Cycle through available shares if more clients than shares
            share_idx = idx % num_shares
            self.client_trigger_map[client_id] = self.trigger_patterns[share_idx]
    
    def init_trigger_masks(self):
        """Pre-create trigger masks for efficient poisoning."""
        dataset = self.params['dataset'].upper()
        img_height, img_width, channels = IMG_SIZE[dataset]
        
        # Create mask for each client
        self.client_masks = {}
        for client_id, pattern in self.client_trigger_map.items():
            mask = torch.zeros((channels, img_height, img_width), device=self.device)
            for x, y in pattern:
                mask[:, x, y] = 1.0
            self.client_masks[client_id] = mask
        
        # Create server mask (combines all patterns)
        self.server_mask = torch.zeros((channels, img_height, img_width), device=self.device)
        for pattern in self.trigger_patterns:
            for x, y in pattern:
                self.server_mask[:, x, y] = 1.0

    def poison_inputs(self, inputs):
        """Apply trigger pattern to inputs based on client ID."""
        poison_inputs = inputs.clone()
        trigger_pixel = 1.0  # White pixel
        
        if self.client_id != -1:
            # Client-side: Apply client-specific trigger pattern
            mask = self.client_masks.get(self.client_id)
            if mask is None:
                # If client is not malicious, return unchanged inputs
                return poison_inputs
            
            # Ensure mask is on the same device as inputs
            if mask.device != inputs.device:
                mask = mask.to(inputs.device)
                self.client_masks[self.client_id] = mask
            
            # Apply mask: set trigger pixels to white
            mask_expanded = mask.unsqueeze(0).expand_as(poison_inputs)
            poison_inputs = torch.where(
                mask_expanded == 1,
                torch.ones_like(poison_inputs) * trigger_pixel,
                poison_inputs
            )
            
            # Save poisoned images if enabled (limit to first few images to avoid too many files)
            if self.save_poisoned_images and self.saved_image_count < 10:
                self._save_images(poison_inputs, f"client_{self.client_id}_pattern")
        else:
            self.server_mask = self.server_mask.to(inputs.device)
            
            mask_expanded = self.server_mask.unsqueeze(0).expand_as(poison_inputs)
            poison_inputs = torch.where(
                mask_expanded == 1,
                torch.ones_like(poison_inputs) * trigger_pixel,
                poison_inputs
            )
            
            # Save poisoned images if enabled (server-side: global trigger)
            if self.save_poisoned_images and self.saved_image_count < 10:
                self._save_images(poison_inputs, "server_global_pattern")
        
        return poison_inputs
    
    def _save_images(self, images, prefix):
        """Save a batch of poisoned images to disk."""
        batch_size = min(images.size(0), 10 - self.saved_image_count)  # Limit total saved images
        
        for i in range(batch_size):
            img_path = os.path.join(self.save_dir, f"{prefix}_img{self.saved_image_count:04d}.png")
            save_image(images[i], img_path, normalize=False)
            self.saved_image_count += 1
            
            if self.saved_image_count >= 10:
                break
        
class Centralized(Distributed):
    """Each client has the same trigger pattern - the aggregated trigger pattern of all shares."""
    
    def poison_inputs(self, inputs):
        """Apply the complete aggregated trigger pattern to inputs."""
        poison_inputs = inputs.clone()
        trigger_pixel = 1.0  # White pixel
        
        # Ensure server mask is on the same device as inputs
        if self.server_mask.device != inputs.device:
            self.server_mask = self.server_mask.to(inputs.device)
        
        # Apply all trigger patterns (server-side aggregated pattern)
        mask_expanded = self.server_mask.unsqueeze(0).expand_as(poison_inputs)
        poison_inputs = torch.where(
            mask_expanded == 1,
            torch.ones_like(poison_inputs) * trigger_pixel,
            poison_inputs
        )
        
        return poison_inputs
