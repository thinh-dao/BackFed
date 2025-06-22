# BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Ray 2.10.0](https://img.shields.io/badge/Ray-2.10.0-blue.svg)](https://docs.ray.io/en/latest/installation.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

BackFed is a comprehensive benchmark framework to efficiently and reliably evaluate backdoor attacks in Federated Learning (FL). This benchmark integrates Ray for parallel execution, Hydra for configuration management, and a modular architecture for easy extension of new attacks and defenses. Compared to existing codebases for backdoor attacks in FL, our framework could **achieve 5X - 10X speedup in training time.**

## Features

- **Modular Architecture**: Easily extend with new attacks, defenses, models, and datasets.
- **Parallel Execution**: Support for both sequential and parallel training modes using Ray with timeout mechanism for client training.
- **Resource Tracking**: Clien-training and server aggregation are monitored based on memory usage and computation time.
- **Comprehensive Attack & Defense Library**: Implementation of various attacks and defenses in a standardized setting for a reliable benchmark.
- **Flexible Configuration**: Hydra-based configuration system for easy experiment setup.
- **Supported Logging**: WandB for real-time visualization and CSV logging for experiment tracking.
- **Resource Management**: Efficient GPU and CPU utilization with configurable resource allocation for parallel execution.

## Installation

### Prerequisites

- Python 3.11
- PyTorch 2.6.0
- Ray 2.10.0
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
  ```bash
  git clone https://github.com/thinh-dao/FL_BackdoorBench.git
  cd FL_BackdoorBench
  ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

```
FL_BackdoorBench/
├── config/                 # Configuration files
│   ├── base.yaml           # Base configuration
│   ├── cifar10.yaml        # CIFAR-10 specific config
│   ├── emnist.yaml         # EMNIST specific config
│   ├── reddit.yaml         # Reddit specific config
│   ├── sentiment140.yaml   # Sentiment140 specific config
│   ├── atk_config/         # Attack configurations
│   └── hydra/              # Hydra configuration
├── backfed/                # Core framework
│   ├── clients/            # Client implementations
│   ├── servers/            # Server implementations
│   ├── poisons/            # Poisoning methods
│   ├── models/             # Model architectures
│   ├── datasets/           # Dataset handling
│   ├── utils/              # Utility functions
│   ├── client_manager.py   # Client management
│   ├── context_actor.py    # Ray context management
│   └── fl_dataloader.py    # Federated data loading
├── experiments/            # Example experiment scripts
├── data/                   # Raw datasets
├── data_splits/            # Pre-computed data partitions
├── checkpoints/            # Model checkpoints
├── csv_results/            # Experiment results
├── figures/                # Generated plots and figures
├── outputs/                # Hydra output logs
├── main.py                 # Main entry point
└──requirements.txt         # Python dependencies
```

## Usage

### Basic Usage

Run a experiment in a no-attack scenario

```bash
python main.py
```

### Customizing Experiments

Modify configuration parameters using Hydra's override syntax:

```bash
python main.py aggregator=unweighted_fedavg dataset=CIFAR10 model=ResNet18 num_rounds=600
```

### Running with Attacks

Enable attacks with specific configurations:

```bash
python main.py aggregator=unweighted_fedavg atk_config=cifar10_multishot atk_config.model_poison_method=base atk_config.data_poison_method=pattern
```

### Running with Defenses

Use a robust aggregation method to defend against attacks:

```bash
python main.py aggregator=trimmed_mean atk_config=cifar10_multishot atk_config.model_poison_method=base atk_config.data_poison_method=pattern
```

## Configuration

The framework uses Hydra for configuration management. Below are the key configuration parameters organized by category:

### Core Experiment Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregator` | str | `unweighted_fedavg` | Aggregation method for federated learning |
| `mode` | str | `parallel` | Training mode (`parallel` or `sequential`) |
| `num_rounds` | int | `600` | Number of federated learning rounds |
| `num_clients` | int | `100` | Total number of clients in the federation |
| `num_clients_per_round` | int | `10` | Number of clients selected per round |
| `seed` | int | `123456` | Random seed for reproducibility |
| `deterministic` | bool | `False` | Enable deterministic training |

### Dataset and Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | - | Dataset name (CIFAR10, CIFAR100, MNIST, EMNIST, etc.) |
| `partitioner` | str | `dirichlet` | Data partitioning method (`uniform` or `dirichlet`) |
| `alpha` | float | `0.5` | Dirichlet distribution parameter for non-IID data |
| `normalize` | bool | `True` | Whether to normalize the dataset |
| `model` | str | - | Model architecture (ResNet18, etc.) |
| `num_classes` | int | - | Number of classes in the dataset |
| `test_batch_size` | int | `512` | Batch size for testing |

### Resource Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda_visible_devices` | str | `1,2,4,5,6` | GPU devices to use |
| `num_cpus` | int | `1` | CPU cores per client |
| `num_gpus` | float | `0.5` | GPU fraction per client |
| `debug` | bool | `False` | Enable debug mode |
| `debug_fraction_data` | float | `0.1` | Fraction of data to use in debug mode |

### Attack Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `no_attack` | bool | `False` | Disable attacks (set to `True` for clean training) |
| `atk_config.model_poison_method` | str | `base` | Model poisoning technique |
| `atk_config.data_poison_method` | str | `pattern` | Data poisoning technique |
| `atk_config.use_atk_optimizer` | bool | `True` | Use separate optimizer for attackers |
| `atk_config.poisoned_lr` | float | `0.05` | Learning rate for poisoned clients |
| `atk_config.poison_epochs` | int | `6` | Training epochs for poisoned clients |
| `atk_config.mutual_dataset` | bool | `False` | Share dataset between attacker and server |
| `atk_config.num_attacker_samples` | int | `640` | Number of clean samples for attacker |

### Federated Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `federated_evaluation` | bool | `False` | Enable federated evaluation |
| `federated_val_split` | float | `0.0` | Validation split for client datasets |

## Attacks and Defenses

### Implemented Attacks

#### Model Poisoning Methods

| Method | Description | Key Parameters | Paper Reference |
|--------|-------------|----------------|-----------------|
| **Base** | Standard model poisoning with direct parameter manipulation | `poisoned_lr`, `poison_epochs` | - |
| **Neurotoxin** | Selective parameter poisoning targeting specific neurons | `neurotoxin_ratio`, `target_layers` | [Neurotoxin](https://arxiv.org/abs/2101.10489) |
| **Chameleon** | Stealthy model poisoning that adapts to defense mechanisms | `adaptive_scaling`, `stealth_factor` | [Chameleon](https://arxiv.org/abs/2108.00888) |

#### Data Poisoning Methods

| Method | Description | Key Parameters | Paper Reference |
|--------|-------------|----------------|-----------------|
| **Pattern** | Adds a visible pattern trigger to images | `pattern_size`, `pattern_location` | [BadNets](https://arxiv.org/abs/1708.06733) |
| **Pixel** | Modifies specific pixels as triggers | `pixel_locations`, `pixel_values` | - |
| **BadNets** | Classic backdoor attack with pattern triggers | `trigger_pattern`, `trigger_size` | [BadNets](https://arxiv.org/abs/1708.06733) |
| **Blended** | Blends a trigger pattern with original images | `blend_ratio`, `trigger_image` | [Blended](https://arxiv.org/abs/1712.05526) |
| **Distributed** | Distributed backdoor attack across multiple clients | `num_trigger_parts`, `part_assignment` | [DBA](https://arxiv.org/abs/1911.07963) |
| **Edge-case** | Targets edge cases in the data distribution | `edge_case_ratio`, `semantic_patterns` | [Edge-case](https://arxiv.org/abs/2007.05084) |
| **A3FL** | Adaptive attack that evolves against defenses | `adaptation_rate`, `strategy_pool` | [A3FL](https://arxiv.org/abs/2106.08814) |
| **IBA** | Input-based backdoor attack with invisible triggers | `trigger_strength`, `frequency_domain` | [IBA](https://arxiv.org/abs/1908.07207) |

### Implemented Defenses

#### Client-Side Defenses

| Defense | Category | Description | Key Parameters | Paper Reference |
|---------|----------|-------------|----------------|-----------------|
| **FedProx** | Client-side | Adds proximal term to client optimization | `mu` (proximal term weight) | [FedProx](https://arxiv.org/abs/1812.06127) |
| **WeakDP** | Client-side | Applies differential privacy at client level | `noise_multiplier`, `max_grad_norm` | [DP-SGD](https://arxiv.org/abs/1607.00133) |

#### Robust Aggregation Defenses

| Defense | Category | Description | Key Parameters | Paper Reference |
|---------|----------|-------------|----------------|-----------------|
| **TrimmedMean** | Robust Aggregation | Removes extreme updates before aggregation | `trim_ratio` | [Byzantine-Robust](https://arxiv.org/abs/1703.02757) |
| **MultiKrum** | Robust Aggregation | Selects subset of updates closest to each other | `krum_k`, `multi_k` | [Krum](https://arxiv.org/abs/1703.02757) |
| **GeometricMedian** | Robust Aggregation | Uses geometric median for aggregation | `max_iterations`, `tolerance` | [Geometric Median](https://arxiv.org/abs/1803.01498) |
| **CoordinateMedian** | Robust Aggregation | Uses coordinate-wise median aggregation | - | [Coordinate-wise](https://arxiv.org/abs/1803.01498) |
| **FLTrust** | Robust Aggregation | Trust-based weighted aggregation with server dataset | `trust_threshold`, `server_data_size` | [FLTrust](https://arxiv.org/abs/2012.13995) |
| **RobustLR** | Robust Aggregation | Adaptive learning rate based on update trustworthiness | `lr_adaptation_factor`, `trust_decay` | - |

#### Anomaly Detection Defenses

| Defense | Category | Description | Key Parameters | Paper Reference |
|---------|----------|-------------|----------------|-----------------|
| **FoolsGold** | Anomaly Detection | Detects sybil attacks via update similarity | `history_length`, `similarity_threshold` | [FoolsGold](https://arxiv.org/abs/1808.04866) |
| **DeepSight** | Anomaly Detection | Clustering-based backdoor detection | `cluster_method`, `anomaly_threshold` | [DeepSight](https://arxiv.org/abs/2201.00763) |
| **RFLBAT** | Anomaly Detection | PCA-based malicious update detection | `pca_components`, `anomaly_ratio` | [RFLBAT](https://arxiv.org/abs/2007.06459) |
| **FLDetector** | Anomaly Detection | Sliding window approach for anomaly detection | `window_size`, `detection_threshold` | [FLDetector](https://arxiv.org/abs/2007.07113) |
| **FLARE** | Anomaly Detection | MMD-based anomaly detection with trust scores | `mmd_threshold`, `trust_alpha` | [FLARE](https://arxiv.org/abs/2201.10025) |
| **Indicator** | Anomaly Detection | Statistical anomaly detection method | `statistical_threshold`, `feature_dims` | - |

#### Hybrid Defenses

| Defense | Category | Description | Key Parameters | Paper Reference |
|---------|----------|-------------|----------------|-----------------|
| **FLAME** | Hybrid | Combines clustering with robust aggregation | `cluster_threshold`, `aggregation_method` | [FLAME](https://arxiv.org/abs/2101.02281) |

### Defense Configuration Examples

Each defense method can be configured with specific parameters. Here are examples:

```bash
# TrimmedMean with 20% trimming
python main.py aggregator=trimmed_mean aggregator_config.trim_ratio=0.2

# FLTrust with server dataset
python main.py aggregator=fltrust aggregator_config.server_data_size=1000

# MultiKrum with k=5
python main.py aggregator=multi_krum aggregator_config.krum_k=5

# FoolsGold with history length 10
python main.py aggregator=foolsgold aggregator_config.history_length=10
```


## Examples

Check the `experiments/` directory for example scripts:

- `clean_training.sh`: Train models without attacks
- `fedavg_vs_attacks.sh`: Evaluate FedAvg against various attacks
- `anomaly_detection.sh`: Test anomaly detection defenses
- `robust_aggregation_multishot.sh`: Test robust aggregation against multishot attacks
- `model_replacement.sh`: Model replacement attack experiments
- `server_lr.sh`: Experiment with different server learning rates
- `sentiment140.sh`: Sentiment140 dataset experiments
- `weakdp_study.sh`: Differential privacy defense studies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Supported Datasets

| Dataset | Type | Classes | Task | Data Splits Available |
|---------|------|---------|------|--------------------|
| **CIFAR10** | Computer Vision | 10 | Image Classification | ✓ |
| **CIFAR100** | Computer Vision | 100 | Image Classification | ✓ |
| **MNIST** | Computer Vision | 10 | Image Classification | ✓ |
| **EMNIST** | Computer Vision | 62 | Image Classification | ✓ |
| **FEMNIST** | Computer Vision | 62 | Image Classification | ✓ |
| **TinyImageNet** | Computer Vision | 200 | Image Classification | ✓ |
| **Reddit** | Natural Language | - | Next Word Prediction | ✓ |
| **Sentiment140** | Natural Language | 2 | Sentiment Classification | ✓ |

### Supported Models

| Model | Domain | Architecture | Compatible Datasets |
|-------|--------|--------------|-------------------|
| **ResNet18** | Computer Vision | Residual Network | CIFAR10, CIFAR100, TinyImageNet |
| **MNISTNet** | Computer Vision | Simple CNN | MNIST, EMNIST, FEMNIST |
| **Simple** | Computer Vision | Basic CNN | CIFAR10, MNIST |
| **Transformer** | Natural Language | Transformer | Reddit, Sentiment140 |
| **WordModel** | Natural Language | LSTM/RNN | Sentiment140 |
| **AutoEncoder** | Computer Vision | Encoder-Decoder | MNIST, CIFAR10 |
| **UNet** | Computer Vision | U-Net | Specialized tasks |

### Available Attack Configurations

| Config Name | Dataset | Attack Type | Description |
|-------------|---------|-------------|-------------|
| `cifar10_multishot` | CIFAR10 | Multi-shot | Multiple rounds of attacks |
| `cifar10_singleshot` | CIFAR10 | Single-shot | One-time attack |
| `emnist_multishot` | EMNIST | Multi-shot | Multiple rounds of attacks |
| `emnist_singleshot` | EMNIST | Single-shot | One-time attack |
| `reddit_multishot` | Reddit | Multi-shot | Multiple rounds of attacks |
| `sentiment140_multishot` | Sentiment140 | Multi-shot | Multiple rounds of attacks |
| `base_attack` | Generic | Configurable | Base attack template |

### Available Aggregation Methods

| Aggregator | Type | Description | Key Parameters |
|------------|------|-------------|----------------|
| `unweighted_fedavg` | Standard | Standard FedAvg aggregation | - |
| `trimmed_mean` | Robust | Trims extreme updates before averaging | `trim_ratio` |
| `multi_krum` | Robust | Selects closest updates using Krum algorithm | `krum_k`, `multi_k` |
| `coordinate_median` | Robust | Uses coordinate-wise median | - |
| `geometric_median` | Robust | Uses geometric median for aggregation | `max_iterations` |
| `fltrust` | Robust | Trust-based aggregation with server data | `server_data_size` |
| `foolsgold` | Anomaly Detection | Sybil attack detection via similarity | `history_length` |
| `deepsight` | Anomaly Detection | Clustering-based backdoor detection | `cluster_method` |
| `rflbat` | Anomaly Detection | PCA-based malicious update detection | `pca_components` |
| `fldetector` | Anomaly Detection | Sliding window anomaly detection | `window_size` |
| `flare` | Anomaly Detection | MMD-based anomaly detection | `mmd_threshold` |
| `indicator` | Anomaly Detection | Statistical anomaly detection | `threshold` |
| `flame` | Hybrid | Clustering + robust aggregation | `cluster_threshold` |
| `fedprox` | Client-side | FedProx with proximal term | `mu` |
| `weakdp` | Client-side | Differential privacy | `noise_multiplier` |
| `robustlr` | Robust | Adaptive learning rate adjustment | `lr_factor` |
