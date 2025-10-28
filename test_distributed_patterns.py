"""
Quick test script to visualize the new distributed trigger patterns.
"""
import torch
import matplotlib.pyplot as plt
from backfed.poisons.distributed import DEFAULT_TRIGGER_PATTERNS

def visualize_patterns(dataset="MNIST"):
    """Visualize trigger patterns for a dataset."""
    patterns = DEFAULT_TRIGGER_PATTERNS[dataset]
    num_shares = len(patterns)
    
    print(f"\n{dataset} Dataset:")
    print(f"Number of Shares: {num_shares}")
    
    # Determine image size based on dataset
    if dataset in ["MNIST", "EMNIST_BYCLASS", "EMNIST_BALANCED", "EMNIST_DIGITS", "FEMNIST"]:
        img_size = (28, 28)
    elif dataset in ["CIFAR10", "CIFAR100"]:
        img_size = (32, 32)
    elif dataset == "TINYIMAGENET":
        img_size = (64, 64)
    
    # Create visualization
    fig, axes = plt.subplots(1, num_shares + 1, figsize=(3 * (num_shares + 1), 3))
    
    # Visualize each share
    for idx, pattern in enumerate(patterns):
        img = torch.zeros(img_size)
        for x, y in pattern:
            img[x, y] = 1.0
        
        axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Share {idx}\n({len(pattern)} pixels)')
        axes[idx].axis('off')
        
        print(f"\nShare {idx} coordinates: {pattern}")
    
    # Visualize aggregated pattern (server-side)
    aggregated = torch.zeros(img_size)
    for pattern in patterns:
        for x, y in pattern:
            aggregated[x, y] = 1.0
    
    axes[-1].imshow(aggregated, cmap='gray', vmin=0, vmax=1)
    axes[-1].set_title(f'Aggregated\n({int(aggregated.sum())} pixels)')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'distributed_patterns_{dataset.lower()}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: distributed_patterns_{dataset.lower()}.png")
    plt.close()

if __name__ == "__main__":
    # Visualize patterns for different datasets
    datasets = ["MNIST", "EMNIST_BYCLASS", "CIFAR10", "TINYIMAGENET"]
    
    for dataset in datasets:
        if dataset in DEFAULT_TRIGGER_PATTERNS:
            visualize_patterns(dataset)
    
    print("\n✓ All visualizations completed!")
