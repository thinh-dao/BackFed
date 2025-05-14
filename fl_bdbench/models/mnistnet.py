import torch.nn as nn
import torch.nn.functional as F
from .simple import SimpleNet

class MnistNet(SimpleNet):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        # Handle RGB inputs (CIFAR10) by converting to grayscale (For BackdoorIndicator defense)
        if x.shape[1] == 3:  # If input has 3 channels (RGB)
            # Convert to grayscale using standard coefficients: 0.299R + 0.587G + 0.114B
            x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x = x.unsqueeze(1)  # Add channel dimension back
        
        # Continue with normal forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        # Use adaptive pooling to handle different input dimensions
        x = self.adaptive_pool(x)
        
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
