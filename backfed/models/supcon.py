########## 
# import sys
# import os

# # Add the parent directory to Python path to find backfed module
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
##########
# Uncomment above to test SupConModel in isolation

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from backfed.models import VGG_CIFAR, VGG_TINYIMAGENET, ResNet_CIFAR, ResNet_MNIST, ResNet_TINYIMAGENET, MnistNet

class SupConModel(nn.Module):
    def __init__(self, model):        
        super(SupConModel, self).__init__()

        if isinstance(model, (VGG_CIFAR, VGG_TINYIMAGENET)):
            # For VGG models (CIFAR and TinyImageNet)
            self.backbone_type = "VGG"
            self.features = copy.deepcopy(model.features)
            
            # For TinyImageNet, also copy the avgpool layer
            if isinstance(model, VGG_TINYIMAGENET):
                self.avgpool = copy.deepcopy(model.avgpool)
            else:
                self.avgpool = None
                
            self.flatten = nn.Flatten()
        elif isinstance(model, (ResNet_CIFAR, ResNet_MNIST, ResNet_TINYIMAGENET)):
            # For CIFAR/MNIST ResNet: explicit layer construction with minimal pooling
            self.backbone_type = "ResNet"
            self.conv1 = copy.deepcopy(model.conv1)
            self.bn1 = copy.deepcopy(model.bn1)
            self.layer1 = copy.deepcopy(model.layer1)
            self.layer2 = copy.deepcopy(model.layer2)
            self.layer3 = copy.deepcopy(model.layer3)
            self.layer4 = copy.deepcopy(model.layer4)
            self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)  # Preserve spatial info
            self.flatten = nn.Flatten()
        elif isinstance(model, MnistNet):
            # For MNISTNet: explicit layer construction
            self.backbone_type = "MnistNet"
            self.conv1 = copy.deepcopy(model.conv1)
            self.conv2 = copy.deepcopy(model.conv2)
            self.adaptive_pool = copy.deepcopy(model.adaptive_pool)
            self.fc1 = copy.deepcopy(model.fc1)  # Feature extractor
            self.flatten = nn.Flatten()
        else:
            raise ValueError("SupConModel currently only supports VGG, ResNet, and MnistNet models")

    def forward(self, x):
        # Extract features using the appropriate backbone
        if self.backbone_type == "VGG":
            # VGG path
            x = self.features(x)
            
            # Apply avgpool if it exists (for TinyImageNet)
            if self.avgpool is not None:
                x = self.avgpool(x)
            
            x = self.flatten(x)
        elif self.backbone_type == "ResNet":
            # ResNet path - explicit layer execution
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.flatten(x)
        elif self.backbone_type == "MnistNet":
            # MNISTNet path - explicit layer execution
            if x.shape[1] == 3:  # Convert RGB to grayscale if needed
                x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
                x = x.unsqueeze(1)  # Add channel dimension back
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = self.adaptive_pool(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
        else:
            raise ValueError("Unknown model architecture")
        
        # Normalize features
        x = F.normalize(x, dim=1)
        return x

    def transfer_params(self, target_model: nn.Module):
        """Transfer compatible parameters from SupConModel to target model."""
        source_params = self.state_dict()
        target_params = target_model.state_dict()
        
        for source_name, source_param in source_params.items():
            # No name mapping needed for explicit layers - they match the original names
            target_name = source_name
            
            if target_name in target_params and target_params[target_name].shape == source_param.shape:
                target_params[target_name].copy_(source_param.clone())


def test_supcon_models():
    """Test function for SupConModel with different architectures."""    
    print("="*60)
    print("Testing SupConModel with different architectures")
    print("="*60)
    
    # Test 1: VGG_CIFAR
    print("\n1. Testing VGG_CIFAR:")
    try:
        from backfed.models.vgg_cifar import VGG_CIFAR
        vgg_cifar = VGG_CIFAR('VGG11', num_classes=10)
        supcon_vgg = SupConModel(vgg_cifar)
        
        # Test forward pass
        x_cifar = torch.randn(2, 3, 32, 32)
        features_vgg = supcon_vgg(x_cifar)
        print(f"   Input shape: {x_cifar.shape}")
        print(f"   Output features shape: {features_vgg.shape}")
        print(f"   Features normalized: {torch.allclose(torch.norm(features_vgg, dim=1), torch.ones(2))}")
        
        # Test parameter transfer
        target_vgg = VGG_CIFAR('VGG11', num_classes=10)
        original_param = target_vgg.features[0].weight.clone()
        supcon_vgg.transfer_params(target_vgg)
        transferred_param = target_vgg.features[0].weight
        print(f"   Parameter transfer successful: {not torch.equal(original_param, transferred_param)}")
        
    except Exception as e:
        print(f"   VGG_CIFAR test failed: {e}")
    
    # Test 2: VGG_TINYIMAGENET
    print("\n2. Testing VGG_TINYIMAGENET:")
    try:
        from backfed.models.vgg_tinyimagenet import VGG as VGG_TINYIMAGENET
        vgg_tiny = VGG_TINYIMAGENET('VGG11', num_classes=200)
        supcon_vgg_tiny = SupConModel(vgg_tiny)
        
        # Test forward pass
        x_tiny = torch.randn(2, 3, 64, 64)
        features_vgg_tiny = supcon_vgg_tiny(x_tiny)
        print(f"   Input shape: {x_tiny.shape}")
        print(f"   Output features shape: {features_vgg_tiny.shape}")
        print(f"   Features normalized: {torch.allclose(torch.norm(features_vgg_tiny, dim=1), torch.ones(2))}")
        
        # Test parameter transfer
        target_vgg_tiny = VGG_TINYIMAGENET('VGG11', num_classes=200)
        original_param = target_vgg_tiny.features[0].weight.clone()
        supcon_vgg_tiny.transfer_params(target_vgg_tiny)
        transferred_param = target_vgg_tiny.features[0].weight
        print(f"   Parameter transfer successful: {not torch.equal(original_param, transferred_param)}")
        
    except Exception as e:
        print(f"   VGG_TINYIMAGENET test failed: {e}")
    
    # Test 3: ResNet_CIFAR
    print("\n3. Testing ResNet_CIFAR:")
    try:
        from backfed.models.resnet_cifar import ResNet18 as ResNet18_CIFAR
        resnet_cifar = ResNet18_CIFAR(num_classes=10)
        supcon_resnet = SupConModel(resnet_cifar)
        
        # Test forward pass
        x_cifar = torch.randn(2, 3, 32, 32)
        features_resnet = supcon_resnet(x_cifar)
        print(f"   Input shape: {x_cifar.shape}")
        print(f"   Output features shape: {features_resnet.shape}")
        print(f"   Features normalized: {torch.allclose(torch.norm(features_resnet, dim=1), torch.ones(2))}")
        
        # Test parameter transfer
        target_resnet = ResNet18_CIFAR(num_classes=10)
        original_param = target_resnet.conv1.weight.clone()
        supcon_resnet.transfer_params(target_resnet)
        transferred_param = target_resnet.conv1.weight
        print(f"   Parameter transfer successful: {not torch.equal(original_param, transferred_param)}")
        
    except Exception as e:
        print(f"   ResNet_CIFAR test failed: {e}")
    
    # Test 4: ResNet_MNIST
    print("\n4. Testing ResNet_MNIST:")
    try:
        from backfed.models.resnet_mnist import ResNet18 as ResNet18_MNIST
        resnet_mnist = ResNet18_MNIST(num_classes=10)
        supcon_mnist = SupConModel(resnet_mnist)
        
        # Test forward pass
        x_mnist = torch.randn(2, 1, 28, 28)
        features_mnist = supcon_mnist(x_mnist)
        print(f"   Input shape: {x_mnist.shape}")
        print(f"   Output features shape: {features_mnist.shape}")
        print(f"   Features normalized: {torch.allclose(torch.norm(features_mnist, dim=1), torch.ones(2))}")
        
        # Test parameter transfer
        target_mnist = ResNet18_MNIST(num_classes=10)
        original_param = target_mnist.conv1.weight.clone()
        supcon_mnist.transfer_params(target_mnist)
        transferred_param = target_mnist.conv1.weight
        print(f"   Parameter transfer successful: {not torch.equal(original_param, transferred_param)}")
        
    except Exception as e:
        print(f"   ResNet_MNIST test failed: {e}")
    
    # Test 5: ResNet_TINYIMAGENET
    print("\n5. Testing ResNet_TINYIMAGENET:")
    try:
        from backfed.models.resnet_tinyimagenet import ResNet18 as ResNet18_TINY
        resnet_tiny = ResNet18_TINY(num_classes=200)
        supcon_resnet_tiny = SupConModel(resnet_tiny)

        x_tiny = torch.randn(2, 3, 64, 64)
        features_resnet_tiny = supcon_resnet_tiny(x_tiny)
        print(f"   Input shape: {x_tiny.shape}")
        print(f"   Output features shape: {features_resnet_tiny.shape}")
        print(
            "   Features normalized: "
            f"{torch.allclose(torch.norm(features_resnet_tiny, dim=1), torch.ones(2))}"
        )

        target_resnet_tiny = ResNet18_TINY(num_classes=200)
        original_param = target_resnet_tiny.conv1.weight.clone()
        supcon_resnet_tiny.transfer_params(target_resnet_tiny)
        transferred_param = target_resnet_tiny.conv1.weight
        print(
            "   Parameter transfer successful: "
            f"{not torch.equal(original_param, transferred_param)}"
        )

    except Exception as e:
        print(f"   ResNet_TINYIMAGENET test failed: {e}")
    
    # Test 6: MnistNet
    print("\n6. Testing MnistNet:")
    try:
        from backfed.models.mnistnet import MnistNet
        mnist_net = MnistNet(num_classes=10)
        supcon_mnistnet = SupConModel(mnist_net)
        
        # Test forward pass with grayscale input
        x_mnist = torch.randn(2, 1, 28, 28)
        features_mnist = supcon_mnistnet(x_mnist)
        print(f"   Input shape (grayscale): {x_mnist.shape}")
        print(f"   Output features shape: {features_mnist.shape}")
        print(f"   Features normalized: {torch.allclose(torch.norm(features_mnist, dim=1), torch.ones(2))}")
        
        # Test forward pass with RGB input (should convert to grayscale)
        x_rgb = torch.randn(2, 3, 28, 28)
        features_rgb = supcon_mnistnet(x_rgb)
        print(f"   Input shape (RGB): {x_rgb.shape}")
        print(f"   Output features shape: {features_rgb.shape}")
        print(f"   Features normalized (RGB): {torch.allclose(torch.norm(features_rgb, dim=1), torch.ones(2))}")
        
        # Test parameter transfer
        target_mnistnet = MnistNet(num_classes=10)
        original_param = target_mnistnet.conv1.weight.clone()
        supcon_mnistnet.transfer_params(target_mnistnet)
        transferred_param = target_mnistnet.conv1.weight
        print(f"   Parameter transfer successful: {not torch.equal(original_param, transferred_param)}")
        
    except Exception as e:
        print(f"   MnistNet test failed: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":        
    test_supcon_models()
