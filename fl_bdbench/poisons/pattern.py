import torch
import random
import torchvision.transforms.v2 as transforms
from PIL import Image
from omegaconf import DictConfig
from typing import Optional, Union
from .base import Poison
from fl_bdbench.const import IMG_SIZE

def pil_to_tensor(img: Image.Image, dataset: str) -> torch.Tensor:
    """Convert PIL Image to tensor with appropriate transformations based on dataset."""
    dataset = dataset.upper()
    if dataset not in DEFAULT_TRANSFORMS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DEFAULT_TRANSFORMS[dataset](img)

# Default transforms for different datasets
DEFAULT_TRANSFORMS = {
    "MNIST": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((28, 28))
    ]),
    "CIFAR10": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((32, 32))
    ]),
    "TINYIMAGENET": transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((64, 64))
    ])
}

# Default trigger patterns
DEFAULT_TRIGGERS = {
    "MNIST": torch.ones((1, 4, 4)), 
    "CIFAR10": torch.ones((3, 4, 4)), 
    "TINYIMAGENET": torch.ones((3, 5, 5))
}

# BadNets trigger weight patterns
BADNETS_TRIGGER_WEIGHTS = {
    "MNIST": torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]),
    "CIFAR10": torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]),
    "TINYIMAGENET": torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
    ])
}

# Add NIST variants to use MNIST patterns
for dataset in ["FEMNIST", "EMNIST_BYCLASS", "EMNIST_BALANCED", "EMNIST_DIGITS"]:
    BADNETS_TRIGGER_WEIGHTS[dataset] = BADNETS_TRIGGER_WEIGHTS["MNIST"]
    DEFAULT_TRIGGERS[dataset] = DEFAULT_TRIGGERS["MNIST"]
    DEFAULT_TRANSFORMS[dataset] = DEFAULT_TRANSFORMS["MNIST"]

# Add CIFAR100 to use CIFAR10 patterns
BADNETS_TRIGGER_WEIGHTS["CIFAR100"] = BADNETS_TRIGGER_WEIGHTS["CIFAR10"]
DEFAULT_TRIGGERS["CIFAR100"] = DEFAULT_TRIGGERS["CIFAR10"]
DEFAULT_TRANSFORMS["CIFAR100"] = DEFAULT_TRANSFORMS["CIFAR10"]

class Pattern(Poison):
    def __init__(self, 
        params: DictConfig,
        client_id: int = -1,
        trigger_pattern: Optional[torch.Tensor] = None, # Backdoor trigger tensor. Shape (1, H, W) or (3, H, W)
        trigger_weight: Optional[Union[torch.Tensor, float]] = None, # Weight mask for the trigger. Shape (1, H, W) or (3, H, W).
        trigger_path: Optional[str] = None, # Path of the trigger image
        location: str = "bottom_right", # 'bottom_left', 'bottom_right', 'top_left', 'top_right', 'center', or 'dynamic'
        x_margin: int = 3, # Margin in x direction
        y_margin: int = 3, # Margin in y direction
        physical_transformation: Optional[transforms.Transform] = None, # Pytorch transformation of the trigger. e.g. rotation, scaling, etc.
    ): 
        super(Pattern, self).__init__(params, client_id)
        self.location = location
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.trigger_path = trigger_path
        self.physical_transformation = physical_transformation
        
        if trigger_path is not None:
            # Load trigger image from file and transform to tensor
            trigger_img = Image.open(self.trigger_path)
            self.trigger_pattern = pil_to_tensor(trigger_img, self.params['dataset'])
        else:
            if trigger_pattern is None:
                dataset = self.params['dataset'].upper()
                self.trigger_pattern = DEFAULT_TRIGGERS[dataset]
            else:
                self.trigger_pattern = trigger_pattern
                
            if self.trigger_pattern.dim() == 2:
                self.trigger_pattern = self.trigger_pattern.unsqueeze(0)

        if trigger_weight is None:
            self.trigger_weight = torch.ones_like(self.trigger_pattern)
        else:
            if isinstance(trigger_weight, float):
                self.trigger_weight = torch.ones_like(self.trigger_pattern) * trigger_weight
            else:
                self.trigger_weight = trigger_weight
            if self.trigger_weight.dim() == 2:
                self.trigger_weight = self.trigger_weight.unsqueeze(0)
        
        if self.trigger_weight.shape[1:] != self.trigger_pattern.shape[1:]:
            raise ValueError("Weight and pattern must have the same shape.")
        
        self.create_img_trigger(self.trigger_pattern, self.trigger_weight)

    def poison_inputs(self, inputs):
        poisoned_inputs = inputs * (1-self.trigger_image_weight) + self.trigger_image * self.trigger_image_weight
        if self.physical_transformation is not None:
            poisoned_inputs = self.physical_transformation(poisoned_inputs)
        return poisoned_inputs
    
    def set_img_trigger(self, trigger_pattern=None, trigger_weight=None):
        if trigger_pattern is None:
            trigger_pattern = self.trigger_pattern
        if trigger_weight is None:
            trigger_weight = self.trigger_weight
        self.create_img_trigger(trigger_pattern, trigger_weight)
        
    def create_img_trigger(self, pattern, weight):
        """
        We cache the trigger image based on the trigger pattern and weight.
        """

        img_height, img_width, img_channels = IMG_SIZE[self.params['dataset'].upper()]
        
        if self.trigger_path is None:
            self.set_trigger_location(pattern, img_height, img_width)
        else:
            self.x_bot, self.y_bot = pattern.shape[1], pattern.shape[2]
            self.x_top, self.y_top = 0, 0

        if self.x_top < 0 or self.y_top < 0 or self.x_bot > img_height or self.y_bot > img_width:
            raise ValueError("The trigger location falls out of the range of the image shape.")
        
        self.trigger_image_weight = torch.zeros((img_channels, img_height, img_width), device=self.device)
        self.trigger_image_weight[:, self.x_top:self.x_bot, self.y_top:self.y_bot] = weight.to(self.device)

        self.trigger_image = torch.zeros((img_channels, img_height, img_width), device=self.device)
        self.trigger_image[:, self.x_top:self.x_bot, self.y_top:self.y_bot] = pattern.to(self.device)

    def set_trigger_location(self, pattern, img_height, img_width):
        if self.location == "dynamic":
            location = random.choice(["top_left", "bottom_left", "top_right", "bottom_right"])
        else:
            location = self.location

        if location == "top_left":
            self.x_top, self.y_top = self.x_margin, self.y_margin
            self.x_bot, self.y_bot = self.x_top + pattern.shape[1], self.y_top + pattern.shape[2]
        elif location == "bottom_left":
            self.x_top, self.y_top = img_height - pattern.shape[1] - self.x_margin - 1, self.y_margin
            self.x_bot, self.y_bot = self.x_top + pattern.shape[1], self.y_top + pattern.shape[2]
        elif location == "top_right":
            self.x_top, self.y_top = self.x_margin, img_width - pattern.shape[2] - self.y_margin - 1
            self.x_bot, self.y_bot = self.x_top + pattern.shape[1], self.y_top + pattern.shape[2]
        elif location == "bottom_right":
            self.x_top, self.y_top = img_height - pattern.shape[1] - self.x_margin - 1, img_width - pattern.shape[2] - self.y_margin - 1
            self.x_bot, self.y_bot = self.x_top + pattern.shape[1], self.y_top + pattern.shape[2]
        else:
            raise ValueError("Invalid location.")

class BadNets(Pattern):
    def __init__(self, *args, **kwargs):
        dataset = kwargs.get('params', {}).get('dataset', '').upper()
        if "NIST" in dataset:
            dataset = "MNIST"
        trigger_weight = BADNETS_TRIGGER_WEIGHTS.get(dataset)
        kwargs['trigger_weight'] = trigger_weight
        super().__init__(*args, **kwargs)
        
class Pixel(Pattern):
    def __init__(self, *args, **kwargs):
        trigger_pattern = torch.ones((1, 1, 1))
        trigger_weight = torch.ones((1, 1, 1))
        kwargs['trigger_pattern'] = trigger_pattern
        kwargs['trigger_weight'] = trigger_weight
        super().__init__(*args, **kwargs)

class Blended(Pattern):
    def __init__(self, *args, **kwargs):
        if kwargs.get('trigger_path') is None:
            kwargs['trigger_path'] = 'fl_bdbench/attack/shared/blended.jpeg'
        if kwargs.get('trigger_weight') is None:
            kwargs['trigger_weight'] = 0.2
        super().__init__(*args, **kwargs)
