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
import hashlib

from omegaconf import OmegaConf
from torch.utils.data import Dataset
from logging import INFO
from backfed.utils import log
from typing import Dict, List, Tuple
from torchvision import datasets
from collections import defaultdict
from tinyimagenet import TinyImageNet

def make_hashable(x):
    if isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple, set)):
        return tuple(make_hashable(i) for i in x)
    return x

def hash_selected_keys(cfg, keys):
    root = OmegaConf.to_container(cfg, resolve=True)
    pairs = []
    for key in keys:
        d, ok = root, True
        for part in key.split('.'):
            if isinstance(d, dict) and part in d:
                d = d[part]
            else:
                ok = False
                break
        if ok:
            pairs.append((key, make_hashable(d)))

    # Use deterministic hash instead of Python's built-in hash()
    content = str(tuple(sorted(pairs))).encode('utf-8')
    return hashlib.md5(content).hexdigest()[:16]  # Use first 16 chars for shorter filename

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
        if "MNIST" in dataset_name:
            self.train_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif "EMNIST" in dataset_name:
            self.train_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomRotation([90, 90]), # Rotate 90 degrees (clockwise or counter-clockwise depends on implementation, but 90 deg rotation is key)
                transforms.RandomHorizontalFlip(p=1.0), # Flip horizontally with p=1.0 (always flip)
            ])
            self.test_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomRotation([90, 90]), # Rotate 90 degrees (clockwise or counter-clockwise depends on implementation, but 90 deg rotation is key)
                transforms.RandomHorizontalFlip(p=1.0), # Flip horizontally with p=1.0 (always flip)
            ])
        elif "FEMNIST" in dataset_name:
            self.train_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.Lambda(lambda x: 1.0 - x),  # Invert the grayscale
                transforms.ToDtype(torch.float32, scale=True),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.Lambda(lambda x: 1.0 - x),  # Invert the grayscale
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name in ["CIFAR10", "CIFAR100"]:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
            self.test_transform = transforms.Compose([ 
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name == "TINYIMAGENET":
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])
        elif dataset_name == "SENTIMENT140" or dataset_name == "REDDIT":
            # No transforms needed for text data
            self.train_transform = None
            self.test_transform = None
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

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

        elif dataset_name == "TINYIMAGENET":
            self.trainset = TinyImageNet(root=datapath, split="train",
                                         transform=self.train_transform)
            self.testset = TinyImageNet(root=datapath, split="val",
                                        transform=self.test_transform)

        elif dataset_name in ["SENTIMENT140", "REDDIT", "FEMNIST"]:
            self.trainset = None
            self.testset = None
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

    def prepare_dataset(self) -> Tuple[Dataset, Dict[int, List[int]], Dict[int, List[int]], Dataset]:
        """
        Distribute the dataset for FL.
        
        Returns
            - trainset: The training dataset
            - client_data_indices: The indices of the training dataset for each participant
            - secret_dataset_indices: The indices of the secret dataset shared by malicious clients
            - testset: The server evaluation dataset
        """
        # Initialize trainset and testset
        self.load_dataset(dataset_name=self.config["dataset"].upper())
        
        if not self.config.debug:
            distribution_keys = ["partitioner"] if self.config.partitioner == "uniform" else ["partitioner", "alpha"]
            keys = ["dataset", *distribution_keys, "num_clients", "seed"]
            if not self.config.no_attack:
                keys.append("atk_config.malicious_clients")
                    
            hash_value = hash_selected_keys(self.config, keys)
            cache_file_path = os.path.join("data_splits", f"{hash_value}.pkl")

        # Try to load from cache if it exists
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'rb') as f:
                    self.client_data_indices = pickle.load(f)
                log(INFO, f"Loaded client data indices from {cache_file_path}")
            except (pickle.PickleError, EOFError) as e:
                log(INFO, f"Error loading cached data split: {e}. Regenerating...")
                os.remove(cache_file_path)  # Remove corrupted cache file
                self._generate_data_split(cache_file_path)
        else:
            # Generate new data split
            self._generate_data_split(cache_file_path)
        
        # Create the secret dataset
        if self.config.atk_config.secret_dataset:
            secret_indices = []
            sample_indices = list(range(len(self.trainset)))
            for no_batch in range(self.config.atk_config.size_of_secret_dataset):
                range_iter = random.sample(sample_indices,
                                        self.config.client_config.batch_size)

                secret_indices.extend(range_iter)
            self.secret_dataset_indices = secret_indices
        else:
            self.secret_dataset_indices = None
            
        return self.trainset, self.client_data_indices, self.secret_dataset_indices, self.testset

    def _sample_dirichlet(self, cids, indices=None) -> Dict[int, List[int]]:
        """
        Dirichlet data distribution for each participant.
        """
        no_participants = len(cids)
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

        for class_idx in class_indices.keys():
            random.shuffle(class_indices[class_idx])
            class_size = len(class_indices[class_idx])
            sampled_probabilities = np.random.dirichlet(
                np.array(no_participants * [self.config.alpha]))
            per_client_size = [round(sampled_probabilities[cid] * class_size) for cid in range(no_participants)]
            random.shuffle(cids)
            
            for idx, cid in enumerate(cids):
                no_imgs = per_client_size[idx] if idx != no_participants - 1 else len(class_indices[class_idx])
                sampled_list = class_indices[class_idx][:no_imgs]
                per_participant_list[cid].extend(sampled_list)
                class_indices[class_idx] = class_indices[class_idx][no_imgs:]

        return per_participant_list

    def _generate_data_split(self, cache_file_path):
        """
        Generate and cache data split for federated learning.

        Args:
            cache_file_path (str): Path to save the cached data split
        """
        sample_indices = list(range(len(self.trainset)))
        sample_cids = list(range(self.config.num_clients))

        # Handle debug mode
        if hasattr(self.config, 'debug') and self.config.debug:
            assert self.config.dataset.upper() not in ["REDDIT", "SENTIMENT140"], "Debug mode only works for CV datasets"
            sample_indices = random.sample(sample_indices, int(self.config.debug_fraction_data * len(sample_indices)))

        # Generate data split based on partitioning strategy
        if self.config.partitioner == "dirichlet":
            self.client_data_indices = self._sample_dirichlet(
                cids=sample_cids,
                indices=sample_indices)
        elif self.config.partitioner == "uniform":
            self.client_data_indices = self._sample_uniform(
                cids=sample_cids,
                indices=sample_indices)
        else:
            raise ValueError(f"Partitioner {self.config.partitioner} is not supported.")
        
        # Cache the generated data split
        try:
            os.makedirs('data_splits', exist_ok=True)
            with open(cache_file_path, 'wb') as f:
                pickle.dump(self.client_data_indices, f)
            log(INFO, f"Cached client data indices to {os.path.basename(cache_file_path)}")
        except Exception as e:
            log(INFO, f"Error caching data split: {e}")

    def _sample_uniform(self, cids, indices=None) -> Dict[int, List[int]]:
        """
        Uniform data distribution for each participant.
        """
        no_participants = len(cids)
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

        for class_idx in class_indices.keys():
            random.shuffle(class_indices[class_idx])
            class_size = len(class_indices[class_idx])
            per_client_size = round(class_size / no_participants)
            random.shuffle(cids)

            for idx, cid in enumerate(cids):
                no_imgs = per_client_size if idx != no_participants - 1 else len(class_indices[class_idx])
                sampled_list = class_indices[class_idx][:no_imgs]
                per_participant_list[cid].extend(sampled_list)
                class_indices[class_idx] = class_indices[class_idx][no_imgs:]

        return per_participant_list

    def visualize_dataset_distribution(self, malicious_clients=None, save_path=None, num_visualized_clients=50):
        log(INFO, f"Visualizing dataset distribution to {save_path}")
        class_counts, indices = FL_DataLoader.get_class_distribution(self.trainset, self.client_data_indices)
        num_classes = len(class_counts)
        num_clients = len(list(class_counts.values())[0])
        df = pd.DataFrame(class_counts, index=indices)

        if num_clients > num_visualized_clients:
            # Always include malicious clients in visualization
            if malicious_clients:
                malicious_set = set(malicious_clients)
                num_malicious = len(malicious_set)
                num_benign_to_sample = num_visualized_clients - num_malicious
                
                if num_benign_to_sample < 0:
                    log(INFO, f"Number of malicious clients ({num_malicious}) exceeds visualization limit ({num_visualized_clients}). Visualizing all malicious clients only.")
                    sampled_indices = list(malicious_set)
                else:
                    # Sample benign clients
                    benign_clients = [cid for cid in range(num_clients) if cid not in malicious_set]
                    sampled_benign = sorted(random.sample(benign_clients, min(num_benign_to_sample, len(benign_clients))))
                    sampled_indices = sorted(list(malicious_set) + sampled_benign)
                    log(INFO, f"Number of clients ({num_clients}) exceeds visualization limit ({num_visualized_clients}). Sampling {len(sampled_benign)} benign clients + {num_malicious} malicious clients for visualization.")
            else:
                log(INFO, f"Number of clients ({num_clients}) exceeds visualization limit ({num_visualized_clients}). Sampling {num_visualized_clients} clients for visualization.")
                sampled_indices = sorted(random.sample(list(range(num_clients)), num_visualized_clients))
            
            df = df.loc[sampled_indices]
            num_clients = len(sampled_indices)

        fig_width = num_clients  # Width scales with the number of clients
        # fig_height = num_classes  # Height scales with the number of classes
        fig_height = 10

        if fig_width > fig_height:
            fig_height = fig_height * 24 / fig_width + 8
            fig_width = 24
            scaling_factor = fig_width / (num_clients ** 0.8)
        else:
            fig_width = fig_width * 24 / fig_height + 8
            fig_height = 24
            scaling_factor = fig_height / (num_classes ** 0.8)

        ax = df.plot(kind='bar', stacked=True, figsize=(fig_width, fig_height), legend=False)

        # Customize the plot with dynamic text sizes
        # plt.title('Per Partition Labels Distribution', fontsize=min(24*scaling_factor, 24))
        # plt.xlabel('Client ID', fontsize=min(20 * scaling_factor, 20))
        # plt.xticks(fontsize=min(16 * scaling_factor, 16))
        # plt.yticks(fontsize=min(16 * scaling_factor, 16))

        plt.title('Client Data Distribution', fontsize=28)
        plt.ylabel('Number of samples', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Change the color of malicious clients
        if malicious_clients:
            xticks = ax.get_xticks()
            for tick_label, tick in zip(ax.get_xticklabels(), xticks):
                if int(tick) in malicious_clients or str(tick) in malicious_clients:
                    tick_label.set_color('red')

        # # Add legend outside the plot on the right with dynamic font size
        # plt.legend(title='Labels', bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.,
        #           fontsize=min(16 * scaling_factor, 16), title_fontsize=min(20 * scaling_factor, 20))

        # Show the plot with tight layout to make room for the legend
        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right side to make room for legend
        plt.tight_layout()

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
            class_indices = sorted(list(dataset.class_to_idx.values()))
        elif hasattr(dataset, 'data') and 'target' in dataset.data.columns:
            # Sentiment140 dataset
            class_indices = sorted(dataset.data['target'].unique().tolist())
        else:
            # Try to infer from the data
            try:
                # Sample a few data points to determine the number of classes
                targets = [dataset[i][1] for i in range(min(100, len(dataset)))]
                if isinstance(targets[0], torch.Tensor):
                    targets = [t.item() for t in targets]
                class_indices = sorted(list(set(targets)))
            except:
                # Fallback to binary classification
                class_indices = [0, 1]

        # Sort client IDs to ensure consistent ordering from 0
        sorted_client_ids = sorted(client_data_indices.keys())
        num_clients = len(sorted_client_ids)
        
        # Initialize counts dictionary with class indices as keys
        class_counts = {idx: [0 for _ in range(num_clients)] for idx in class_indices}

        # Count samples per class per client  
        for client_idx, (client_id, data_indices) in enumerate(sorted(client_data_indices.items())):
            for idx in data_indices:
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
                    class_counts[target][client_idx] += 1

        return class_counts, sorted_client_ids
