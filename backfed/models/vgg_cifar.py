'''
VGG implementation for CIFAR-10/100 datasets.
Based on: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''

import math
import torch
import torch.nn as nn

__all__ = [
    'VGG', 'vgg9', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'get_vgg_model'
]

# Configuration for different VGG variants
cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    '''
    Unified VGG model supporting all variants with configurable classifiers.
    
    Args:
        vgg_name: VGG variant ('VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19')
        num_classes: Number of output classes (default: 10)
        batch_norm: Whether to use batch normalization (default: False)
        classifier_type: 'simple' for single linear layer, 'mlp' for 3-layer MLP (default: 'simple')
        dropout: Dropout rate for MLP classifier (default: 0.5)
    '''
    def __init__(self, vgg_name='VGG11', num_classes=10, batch_norm=False, 
                 classifier_type='simple', dropout=0.5):
        super(VGG, self).__init__()
        
        if vgg_name not in cfg:
            raise ValueError(f"Unknown VGG variant: {vgg_name}. Choose from {list(cfg.keys())}")
        
        self.vgg_name = vgg_name
        self.num_classes = num_classes
        self.classifier_type = classifier_type
        
        # Build feature extractor
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        
        # Build classifier based on type
        if classifier_type == 'simple':
            # Single linear layer for VGG9
            self.classifier = nn.Linear(512, num_classes)
        elif classifier_type == 'mlp':
            # 3-layer MLP classifier
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Choose 'simple' or 'mlp'")
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        # Add final pooling for VGG9 compatibility
        if self.vgg_name == 'VGG9':
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# Factory functions
def vgg9(num_classes=10, batch_norm=False):
    """VGG 9-layer model with simple classifier"""
    return VGG('VGG9', num_classes=num_classes, batch_norm=batch_norm, classifier_type='simple')

def vgg11(num_classes=10, batch_norm=False, classifier_type='mlp'):
    """VGG 11-layer model"""
    return VGG('VGG11', num_classes=num_classes, batch_norm=batch_norm, classifier_type=classifier_type)

def vgg11_bn(num_classes=10, classifier_type='mlp'):
    """VGG 11-layer model with batch normalization"""
    return VGG('VGG11', num_classes=num_classes, batch_norm=True, classifier_type=classifier_type)

def vgg13(num_classes=10, batch_norm=False, classifier_type='mlp'):
    """VGG 13-layer model"""
    return VGG('VGG13', num_classes=num_classes, batch_norm=batch_norm, classifier_type=classifier_type)

def vgg13_bn(num_classes=10, classifier_type='mlp'):
    """VGG 13-layer model with batch normalization"""
    return VGG('VGG13', num_classes=num_classes, batch_norm=True, classifier_type=classifier_type)

def vgg16(num_classes=10, batch_norm=False, classifier_type='mlp'):
    """VGG 16-layer model"""
    return VGG('VGG16', num_classes=num_classes, batch_norm=batch_norm, classifier_type=classifier_type)

def vgg16_bn(num_classes=10, classifier_type='mlp'):
    """VGG 16-layer model with batch normalization"""
    return VGG('VGG16', num_classes=num_classes, batch_norm=True, classifier_type=classifier_type)

def vgg19(num_classes=10, batch_norm=False, classifier_type='mlp'):
    """VGG 19-layer model"""
    return VGG('VGG19', num_classes=num_classes, batch_norm=batch_norm, classifier_type=classifier_type)

def vgg19_bn(num_classes=10, classifier_type='mlp'):
    """VGG 19-layer model with batch normalization"""
    return VGG('VGG19', num_classes=num_classes, batch_norm=True, classifier_type=classifier_type)

def test():
    net = vgg11()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    
def get_cifar_vgg_model(vgg_name, num_classes=10, batch_norm=True, classifier_type=None):
    """
    Factory function for backward compatibility with existing code.
    
    Args:
        vgg_name: 'vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'
        num_classes: Number of output classes
        batch_norm: Whether to use batch normalization
        classifier_type: 'simple' or 'mlp'. If None, uses defaults (simple for VGG9, mlp for others)
    """
    
    vgg_name_upper = vgg_name.upper()
    if vgg_name_upper not in cfg:
        raise ValueError(f"Unknown VGG variant: {vgg_name}. Choose from {[k.lower() for k in cfg.keys()]}")
    
    # Set default classifier type if not specified
    if classifier_type is None:
        classifier_type = 'simple' if vgg_name_upper == 'VGG9' else 'mlp'
    
    return VGG(vgg_name_upper, num_classes=num_classes, batch_norm=batch_norm, classifier_type=classifier_type)