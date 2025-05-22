"""
Processing and distributing datasets for FL.
"""

import torch
import torchvision.transforms.v2 as transforms
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from backfed.datasets import (
    FEMNIST,
    load_sentiment140_for_albert,
    load_sentiment140_for_lstm,
    sentiment140_collate_fn,
    load_reddit_for_lstm
)
from torch.utils.data import Dataset, DataLoader
from logging import INFO
from backfed.utils import log
from typing import Dict, List, Tuple
from torchvision import datasets
from collections import defaultdict
from tinyimagenet import TinyImageNet

class FL_DataLoader:
    """
    Federated Learning DataLoader for multiple datasets.
    Handles dataset loading, partitioning, and transformations for federated settings.
    """

    def __init__(self, config):
        """
        Initialize the FL_DataLoader with the given configuration.
        Args:
            config (dict): Configuration dictionary containing dataset and training parameters.
        """
        self.config = config
        dataset_name = self.config["dataset"].upper()

        # Define standard transformations
        if "NIST" in dataset_name:
            self.train_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name in ["CIFAR10", "CIFAR100"]:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name == "TINYIMAGENET":
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name == "SENTIMENT140" or dataset_name == "REDDIT":
            # No transforms needed for text data
            self.train_transform = None
            self.test_transform = None
            self.load_dataset(dataset_name)
            return
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        # Extract the last three transforms for test set
        self.test_transform = transforms.Compose(self.train_transform.transforms[-3:])
        self.load_dataset(dataset_name)

    def load_dataset(self, dataset_name):
        """
        Load the specified dataset and apply transformations.
        Args:
            dataset_name (str): Name of the dataset to load.
        """
        datapath = os.path.join(self.config["datapath"], dataset_name)

        if dataset_name == "CIFAR10":
            self.trainset = datasets.CIFAR10(datapath, train=True, download=True,
                                             transform=self.train_transform)
            self.testset = datasets.CIFAR10(datapath, train=False, download=True,
                                            transform=self.test_transform)

        elif dataset_name == "CIFAR100":
            self.trainset = datasets.CIFAR100(datapath, train=True, download=True,
                                              transform=self.train_transform)
            self.testset = datasets.CIFAR100(datapath, train=False, download=True,
                                             transform=self.test_transform)

        elif dataset_name == "MNIST":
            self.trainset = datasets.MNIST(datapath, train=True, download=True,
                                           transform=self.train_transform)
            self.testset = datasets.MNIST(datapath, train=False, download=True,
                                          transform=self.test_transform)

        elif "EMNIST" in dataset_name:
            split = dataset_name.split("_")[-1].lower()
            datapath = os.path.join(self.config["datapath"], "EMNIST")
            self.trainset = datasets.EMNIST(datapath, train=True, split=split, download=True,
                                            transform=self.train_transform)
            self.testset = datasets.EMNIST(datapath, train=False, split=split, download=True,
                                           transform=self.test_transform)
            if self.trainset.split == self.testset.split == "letters":
                self.trainset.targets -= 1
                self.testset.targets -= 1

        elif dataset_name == "FEMNIST":
            self.trainset = FEMNIST(datapath, train=True, download=True,
                                    transform=self.train_transform)
            self.testset = FEMNIST(datapath, train=False, download=True,
                                   transform=self.test_transform)

        elif dataset_name == "TINYIMAGENET":
            self.trainset = TinyImageNet(root=datapath, split="train",
                                         transform=self.train_transform)
            self.testset = TinyImageNet(root=datapath, split="val",
                                        transform=self.test_transform)

        elif dataset_name == "SENTIMENT140":
            # Check if we're using a supported model
            supported_models = ["albert", "lstm"]
            assert self.config["model"].lower() in supported_models, f"Model {self.config['model']} is not supported for Sentiment140 dataset. Supported models are {supported_models}"

            if self.config["model"].lower() == "albert":
                self.trainset, self.testset = load_sentiment140_for_albert(
                    config=self.config,
                )

            else:
                self.trainset, self.testset = load_sentiment140_for_lstm(
                    config=self.config,
                )

        elif dataset_name == "REDDIT":
            # Check if we're using a supported model
            supported_models = ["lstm"]
            assert self.config["model"].lower() in supported_models, f"Model {self.config['model']} is not supported for Reddit dataset. Supported models are {supported_models}"
            
            self.trainset, self.testset = load_reddit_for_lstm(self.config)

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        if self.config.atk_config.data_poison_method == "semantic":
            assert dataset_name == "CIFAR10", "Semantic attack is only available for CIFAR10"
            self.prepare_cifar10_semantic()

        return True


    def prepare_cifar10_semantic(self, semantic_path='./data/semantic/cifar10_semantic_car.pkl'):
        """
        Prepare CIFAR10 semantic poisoning samples and remove them from the training set.
        Args:
            semantic_path (str): Path to save the poisoned images.
        """
        poison_images_index_stripe = [2180, 2771, 3233, 4932, 6241, 6813, 6869, 9476, 11395, 11744, 14209, 14238, 18716, 19793, 20781, 21529, 31311, 40518, 40633, 42119, 42663, 49392]
        poison_images_index_green = [389, 561, 874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941, 34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861, 47001, 47026, 48003, 48030, 49163, 49588]
        poison_images_index_wall = [330, 568, 3934, 12336, 30560, 30696, 33105, 33615, 33907, 36848, 40713, 41706]
        poison_index_cars = poison_images_index_stripe + poison_images_index_green + poison_images_index_wall

        # Dump all images from poison_index_cars to cifar10_semantic_car.pkl
        if not os.path.isfile(semantic_path):
            os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
            # Retrieve poison images from the trainset
            poison_images = [self.trainset[idx] for idx in poison_index_cars]
            import pickle
            with open(semantic_path, 'wb') as f:
                pickle.dump(poison_images, f)
            log(INFO, f"Dumped {len(poison_images)} poison images to {semantic_path}")

        # Remove poison_index_cars samples from self.train_dataset
        self.trainset = torch.utils.data.Subset(self.trainset, [i for i in range(len(self.trainset)) if i not in poison_index_cars])

    def prepare_dataset(self) -> Tuple[Dataset, Dict[int, List[int]], DataLoader]:
        """
        Distribute the dataset for FL.
        Returns:
            trainset: The training dataset
            client_data_indices: The indices of the training dataset for each participant
            test_loader: The test loader
        """
        # Create directory for caching data splits
        os.makedirs("data_splits", exist_ok=True)

        # Create a unique filename based on configuration parameters
        load_split_file = f"{self.config.dataset}_{self.config.num_clients}_{self.config.no_attack}_{self.config.atk_config.mutual_dataset}_{self.config.atk_config.num_attacker_samples}"
        load_split_file += f"_{self.config.partitioner}" if self.config.partitioner == "uniform" else f"_{self.config.partitioner}_{self.config.alpha}"

        # Add debug info to filename if in debug mode
        if hasattr(self.config, 'debug') and self.config.debug:
            load_split_file += f"_debug={self.config.debug}_fraction={self.config.debug_fraction_data}"

        # Add random seed to filename if available for reproducibility
        if hasattr(self.config, 'seed'):
            load_split_file += f"_seed={self.config.seed}"

        # Full path to the cache file
        cache_file_path = os.path.join("data_splits", load_split_file + ".pkl")

        # Try to load from cache if it exists
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'rb') as f:
                    self.client_data_indices = pickle.load(f)
                log(INFO, f"Loaded client data indices from {load_split_file + '.pkl'}")
            except (pickle.PickleError, EOFError) as e:
                log(INFO, f"Error loading cached data split: {e}. Regenerating...")
                os.remove(cache_file_path)  # Remove corrupted cache file
                self._generate_data_split(cache_file_path)
        else:
            # Generate new data split
            self._generate_data_split(cache_file_path)

        # Server-side test loader (for server-side evaluation)
        if self.config.dataset.upper() == "SENTIMENT140":
            self.test_loader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=self.config.test_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                shuffle=False,
                collate_fn=sentiment140_collate_fn
            )
        else:
            self.test_loader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=self.config.test_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                shuffle=False
            )

        return self.trainset, self.client_data_indices, self.test_loader

    def _sample_dirichlet(self, no_participants, indices=None) -> Dict[int, List[int]]:
        """
        Dirichlet data distribution for each participant.
        """
        if indices is None:
            indices = list(range(len(self.trainset)))  # Sample all the indices

        log(INFO, f"Sampling train dataset ({len(indices)} samples) for {no_participants} partitions with Dirichlet distribution (alpha={self.config.alpha}).")

        class_indices = {}

        # Handle different dataset types
        if hasattr(self.trainset, 'targets'):
            # Standard torchvision datasets
            for ind in indices:
                label = self.trainset.targets[ind]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]
        elif hasattr(self.trainset, 'data') and hasattr(self.trainset.data, 'target'):
            # Sentiment140 dataset
            for ind in indices:
                label = self.trainset.data.iloc[ind]['target']
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]
        else:
            # Try to handle generic datasets
            for ind in indices:
                _, label = self.trainset[ind]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]

        per_participant_list = defaultdict(list)
        clients = list(range(no_participants))

        for class_idx in class_indices.keys():
            random.shuffle(class_indices[class_idx])
            class_size = len(class_indices[class_idx])
            sampled_probabilities = np.random.dirichlet(
                np.array(no_participants * [self.config.alpha]))
            per_client_size = [round(sampled_probabilities[cid] * class_size) for cid in range(no_participants)]
            random.shuffle(clients)

            for cid in range(len(clients)):
                no_imgs = per_client_size[cid] if cid != no_participants - 1 else len(class_indices[class_idx])
                sampled_list = class_indices[class_idx][:no_imgs]
                per_participant_list[clients[cid]].extend(sampled_list)
                class_indices[class_idx] = class_indices[class_idx][no_imgs:]

        return per_participant_list

    def _generate_data_split(self, cache_file_path):
        """
        Generate and cache data split for federated learning.

        Args:
            cache_file_path (str): Path to save the cached data split
        """
        # Determine which samples to use
        if not self.config.no_attack and self.config.atk_config.mutual_dataset:
            # Split the dataset into two subsets: clean and attacker-controlled samples
            indices = list(range(len(self.trainset)))
            attacker_indices = np.random.choice(indices, self.config.atk_config.num_attacker_samples, replace=False)
            sample_indices = [i for i in indices if i not in attacker_indices]
        else:
            attacker_indices = None
            sample_indices = list(range(len(self.trainset)))

        # Handle debug mode
        if hasattr(self.config, 'debug') and self.config.debug:
            assert self.config.dataset.upper() not in ["REDDIT", "SENTIMENT140"], "Debug mode only works for CV datasets"
            sample_indices = np.random.choice(sample_indices, int(self.config.debug_fraction_data * len(sample_indices)), replace=False)

        # Generate data split based on partitioning strategy
        if self.config.partitioner == "dirichlet":
            self.client_data_indices = self._sample_dirichlet(
                no_participants=self.config.num_clients,
                indices=sample_indices)
        elif self.config.partitioner == "uniform":
            self.client_data_indices = self._sample_uniform(
                no_participants=self.config.num_clients,
                indices=sample_indices)
        else:
            raise ValueError(f"Partitioner {self.config.partitioner} is not supported.")

        # Cache the generated data split
        try:
            with open(cache_file_path, 'wb') as f:
                pickle.dump(self.client_data_indices, f)
            log(INFO, f"Cached client data indices to {os.path.basename(cache_file_path)}")
        except Exception as e:
            log(INFO, f"Error caching data split: {e}")

    def _sample_uniform(self, no_participants, indices=None) -> Dict[int, List[int]]:
        """
        Uniform data distribution for each participant.
        """
        if indices is None:
            indices = list(range(len(self.trainset)))  # Sample all the indices

        log(INFO, f"Sampling train dataset ({len(indices)} samples) uniformly for {no_participants} partitions.")

        class_indices = {}

        # Handle different dataset types
        if hasattr(self.trainset, 'targets'):
            # Standard torchvision datasets
            for ind in indices:
                label = self.trainset.targets[ind]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]
        elif hasattr(self.trainset, 'data') and hasattr(self.trainset.data, 'target'):
            # Sentiment140 dataset
            for ind in indices:
                label = self.trainset.data.iloc[ind]['target']
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]
        else:
            # Try to handle generic datasets
            for ind in indices:
                _, label = self.trainset[ind]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label in class_indices:
                    class_indices[label].append(ind)
                else:
                    class_indices[label] = [ind]

        per_participant_list = defaultdict(list)
        clients = list(range(no_participants))

        for class_idx in class_indices.keys():
            random.shuffle(class_indices[class_idx])
            class_size = len(class_indices[class_idx])
            per_client_size = round(class_size / no_participants)
            random.shuffle(clients)

            for cid in range(len(clients)):
                no_imgs = per_client_size if cid != no_participants - 1 else len(class_indices[class_idx])
                sampled_list = class_indices[class_idx][:no_imgs]
                per_participant_list[clients[cid]].extend(sampled_list)
                class_indices[class_idx] = class_indices[class_idx][no_imgs:]

        return per_participant_list

    def visualize_dataset_distribution(self, malicious_clients=None, save_path=None):
        log(INFO, f"Visualizing dataset distribution to {save_path}")
        class_counts, indices = FL_DataLoader.get_class_distribution(self.trainset, self.client_data_indices)
        num_classes = len(class_counts)
        num_clients = len(list(class_counts.values())[0])
        df = pd.DataFrame(class_counts, index=indices)

        fig_width = num_clients  # Width scales with the number of clients
        fig_height = num_classes  # Height scales with the number of classes

        if fig_width > fig_height:
            fig_height = fig_height * 24 / fig_width + 8
            fig_width = 24
            scaling_factor = fig_width / (num_clients ** 0.8)
        else:
            fig_width = fig_width * 24 / fig_height + 8
            fig_height = 24
            scaling_factor = fig_height / (num_classes ** 0.8)

        ax = df.plot(kind='bar', stacked=True, figsize=(fig_width, fig_height))

        # Customize the plot with dynamic text sizes
        plt.title('Per Partition Labels Distribution', fontsize=min(24*scaling_factor, 24))
        plt.xlabel('Client ID', fontsize=min(20 * scaling_factor, 20))
        plt.ylabel('Number of samples', fontsize=min(20 * scaling_factor, 20))
        plt.xticks(fontsize=min(16 * scaling_factor, 16))
        plt.yticks(fontsize=min(16 * scaling_factor, 16))

        # Change the color of malicious clients
        if malicious_clients:
            xticks = ax.get_xticks()
            for tick_label, tick in zip(ax.get_xticklabels(), xticks):
                if int(tick) in malicious_clients or str(tick) in malicious_clients:
                    tick_label.set_color('red')

        # Add legend outside the plot on the right with dynamic font size
        plt.legend(title='Labels', bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.,
                  fontsize=min(16 * scaling_factor, 16), title_fontsize=min(20 * scaling_factor, 20))

        # Show the plot with tight layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right side to make room for legend

        if save_path:
            path = os.path.join(save_path, f"data_distribution.pdf")
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    @staticmethod
    def get_class_distribution(dataset, client_data_indices):
        """
        Get the distribution of classes across clients using class indices as keys.

        Args:
            dataset: The dataset
            client_data_indices: Dict of client_id -> data indices

        Returns:
            class_counts: Dictionary mapping class indices to counts per client
            client_ids: Range of client IDs
        """
        # Determine the number of classes
        if hasattr(dataset, 'class_to_idx'):
            # Standard torchvision datasets
            class_indices = list(dataset.class_to_idx.values())
        elif hasattr(dataset, 'data') and 'target' in dataset.data.columns:
            # Sentiment140 dataset
            class_indices = dataset.data['target'].unique().tolist()
        else:
            # Try to infer from the data
            try:
                # Sample a few data points to determine the number of classes
                targets = [dataset[i][1] for i in range(min(100, len(dataset)))]
                if isinstance(targets[0], torch.Tensor):
                    targets = [t.item() for t in targets]
                class_indices = list(set(targets))
            except:
                # Fallback to binary classification
                class_indices = [0, 1]

        # Initialize counts dictionary with class indices as keys
        class_counts = {idx: [0 for _ in range(len(client_data_indices))] for idx in class_indices}

        # Count samples per class per client
        for client_id, client_idx in client_data_indices.items():
            for idx in client_idx:
                # Get the target based on the dataset type
                if hasattr(dataset, 'targets'):
                    # Standard torchvision datasets
                    target = dataset.targets[idx]
                    if isinstance(target, torch.Tensor):
                        target = target.item()
                elif hasattr(dataset, 'data') and 'target' in dataset.data.columns:
                    # Sentiment140 dataset
                    target = dataset.data.iloc[idx]['target']
                else:
                    # Try to get the target directly from the dataset
                    _, target = dataset[idx]
                    if isinstance(target, torch.Tensor):
                        target = target.item()

                if target in class_counts:
                    class_counts[target][client_id] += 1

        return class_counts, client_data_indices.keys()
