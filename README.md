# BackFed: An Efficient Benchmark for Backdoor Attacks in Federated Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Ray 2.10.0](https://img.shields.io/badge/Ray-2.10.0-blue.svg)](https://docs.ray.io/en/latest/installation.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

BackFed is a comprehensive framework for evaluating backdoor attacks and defenses in federated learning environments. This benchmark provides an efficient and standardized way to implement, test, and compare various attack and defense mechanisms in federated learning settings.

## Features

- **Modular Architecture**: Easily extend with new attacks, defenses, models, and datasets
- **Parallel Execution**: Support for both sequential and parallel training modes using Ray
- **Comprehensive Attack Library**: Implementation of various backdoor attacks including:
  - Model poisoning methods: Base, Neurotoxin, Chameleon, 3DFed
  - Data poisoning methods: Pattern, Pixel, BadNets, Blended, Distributed, Edge-case, A3FL, IBA
- **Robust Defense Mechanisms**: Multiple defense strategies including:
  - FedAvg, TrimmedMean, MultiKrum, FedProx, FLAME, FoolsGold, WeakDP, DeepSight, RFLBAT, FLTrust, FLARE
- **Flexible Configuration**: Hydra-based configuration system for easy experiment setup
- **Detailed Logging**: Support for WandB and CSV logging for experiment tracking
- **Resource Management**: Efficient GPU and CPU utilization with configurable resource allocation

## Installation

### Prerequisites

- Python 3.11
- PyTorch 2.6.0
- Ray 2.10.0
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BackFed.git
   cd BackFed
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
BackFed/
├── config/                 # Configuration files
│   ├── defaults.yaml       # Default configuration
│   └── atk_config/         # Attack configurations
├── fl_bdbench/             # Core framework
│   ├── clients/            # Client implementations
│   ├── servers/            # Server implementations
│   ├── poisons/            # Poisoning methods
│   ├── models/             # Model architectures
│   └── utils/              # Utility functions
├── experiments/            # Example experiment scripts
├── main.py                 # Main entry point
└── README.md               # This file
```

## Usage

### Basic Usage

Run a federated learning experiment with default settings:

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
python main.py aggregator=unweighted_fedavg atk_config=multishot atk_config.model_poison_method=base atk_config.data_poison_method=pattern
```

### Running with Defenses

Use a robust aggregation method to defend against attacks:

```bash
python main.py aggregator=trimmed_mean atk_config=multishot atk_config.model_poison_method=base atk_config.data_poison_method=pattern
```

## Configuration

The framework uses Hydra for configuration management. Key configuration options include:

### General Settings

- `aggregator`: Aggregation method (e.g., unweighted_fedavg, trimmed_mean)
- `mode`: Training mode (parallel or sequential)
- `num_rounds`: Number of federated learning rounds
- `num_clients`: Total number of clients
- `num_clients_per_round`: Number of clients selected per round

### Dataset and Model

- `dataset`: Dataset to use (CIFAR10, CIFAR100, MNIST, etc.)
- `model`: Model architecture (ResNet18, etc.)
- `partitioner`: Data partitioning method (uniform, dirichlet)

### Attack Settings

- `no_attack`: Set to False to enable attacks
- `atk_config.model_poison_method`: Model poisoning method
- `atk_config.data_poison_method`: Data poisoning method
- `atk_config.poison_rate`: Portion of data to poison
- `atk_config.malicious_clients`: List of malicious client IDs

### Defense Settings

Each defense has its own configuration parameters in the `aggregator_config` section.

## Attacks and Defenses

### Implemented Attacks

#### Model Poisoning Methods
- **Base**: Standard model poisoning
- **Neurotoxin**: Selective parameter poisoning
- **Chameleon**: Stealthy model poisoning
- **3DFed**: Three-dimensional attack on federated learning

#### Data Poisoning Methods
- **Pattern**: Adds a visible pattern to images
- **Pixel**: Modifies specific pixels
- **BadNets**: Backdoor attack with pattern triggers
- **Blended**: Blends a trigger pattern with the original image
- **Distributed**: Distributed backdoor attack
- **Edge-case**: Targets edge cases in the data distribution
- **A3FL**: Adaptive attack on federated learning
- **IBA**: Input-based backdoor attack

### Implemented Defenses

Our framework provides various defense mechanisms categorized as follows:

#### Client-Side Defenses
These defenses operate during client training by modifying the client's training process:
- **FedProx**: Adds a proximal term to client optimization
- **WeakDP**: Applies differential privacy at the client level

#### Robust Aggregation Defenses
These defenses modify the aggregation algorithm to be resilient against malicious updates:
- **TrimmedMean**: Removes extreme updates from client updates
- **MultiKrum**: Selects a subset of client updates that are closest to each other
- **GeometricMedian**: Uses geometric median for aggregation
- **CoordinateMedian**: Uses coordinate-wise median for aggregation
- **NormClipping**: Clips client updates to a maximum norm
- **FLTrust**: Uses a trusted dataset to assign trust scores for weighted aggregation
- **RobustLR**: Adjusts learning rates based on client update trustworthiness

#### Anomaly Detection Defenses
These defenses identify and filter malicious updates by detecting statistical anomalies:
- **FoolsGold**: Identifies sybil attacks by detecting similar updates
- **DeepSight**: Uses clustering-based approach to detect backdoor attacks
- **RFLBAT**: Uses PCA-based detection of malicious updates
- **FLDetector**: Uses a sliding window approach to detect anomalies
- **FLARE**: Uses Maximum Mean Discrepancy (MMD) to detect anomalies and assign trust scores

#### Hybrid Defenses
These defenses combine techniques from multiple categories:
- **FLAME**: Combines anomaly detection (clustering) with robust aggregation techniques

## Examples

Check the `experiments/` directory for example scripts:

- `clean_training.sh`: Train models without attacks
- `fedavg_vs_attacks.sh`: Evaluate FedAvg against various attacks
- `server_lr.sh`: Experiment with different server learning rates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
