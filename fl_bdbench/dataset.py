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

from torch.utils.data import Dataset, DataLoader
from logging import INFO
from fl_bdbench.utils import log
from typing import Dict, List, Tuple
from torchvision import datasets
from collections import defaultdict
from PIL import Image
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
            print(f"Dumped {len(poison_images)} poison images to {semantic_path}")

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
        if not self.config.no_attack and self.config.atk_config.mutual_dataset:
            # Split the dataset into two subsets: clean and attacker-controlled samples
            indices = list(range(len(self.trainset)))
            attacker_indices = np.random.choice(indices, self.config.atk_config.num_attacker_samples, replace=False)
            sample_indices = [i for i in indices if i not in attacker_indices]
        else:
            attacker_indices = None
            sample_indices = list(range(len(self.trainset)))

        if self.config.debug:
            sample_indices = np.random.choice(sample_indices, int(self.config.debug_fraction_data * len(sample_indices)), replace=False)

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

        # Server-side test loader (for server-side evaluation)
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
        for ind in indices:
            label = self.trainset.targets[ind]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in class_indices:
                class_indices[label].append(ind)
            else:
                class_indices[label] = [ind]

        per_participant_list = defaultdict(list)
        num_classes = len(class_indices.keys())
        clients = list(range(no_participants))

        for class_idx in range(num_classes):
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

    def _sample_uniform(self, no_participants, indices=None) -> Dict[int, List[int]]:
        """
        Uniform data distribution for each participant.
        """
        if indices is None:
            indices = list(range(len(self.trainset)))  # Sample all the indices

        log(INFO, f"Sampling train dataset ({len(indices)} samples) uniformly for {no_participants} partitions.")
        
        class_indices = {}
        for ind in indices:
            label = self.trainset.targets[ind]
            if label in class_indices:
                class_indices[label].append(ind)
            else:
                class_indices[label] = [ind]

        per_participant_list = defaultdict(list)
        num_classes = len(class_indices.keys())
        clients = list(range(no_participants))

        for class_idx in range(num_classes):
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


    def save_visualization(self, malicious_clients=None, save_path=None):
        """
        Visualize and save the dataset distribution for train/val splits.
        """
        log(INFO, f"Visualizing dataset distribution to {save_path}")
        self.visualize_dataset_distribution(malicious_clients, save_path, split="train")
        if self.config.val_split > 0:
            self.visualize_dataset_distribution(malicious_clients, save_path, split="val")


    def visualize_dataset_distribution(self, malicious_clients=None, save_path=None, split="train"):
        if split == "train":
            datasets = self.train_datasets
        elif split == "val":
            datasets = self.val_datasets
        else:
            raise ValueError(f"Split {split} is not distributed.")
        
        class_counts, indices = FL_DataLoader.get_class_distribution(datasets)
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
            path = os.path.join(save_path, f"{split}_data_distribution.pdf")
            plt.savefig(path, dpi=500, bbox_inches='tight')
        else:
            plt.show()


    @staticmethod   
    def get_class_distribution(datasets):
        num_clients = len(datasets)
        temp_dataset = datasets[0]
        while isinstance(temp_dataset, torch.utils.data.dataset.Subset):
            temp_dataset = temp_dataset.dataset

        idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}
        class_counts = {idx_to_class[idx]: [0 for _ in range(num_clients)] for idx in idx_to_class}

        indices = list(range(num_clients))
        for client_id, dataset in datasets.items():
            for _, label in dataset:
                class_counts[idx_to_class[label]][client_id] += 1

        return class_counts, indices


# Taken from https://github.com/tao-shen/FEMNIST_pytorch/blob/master/femnist.py
from torchvision.datasets import MNIST, utils

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)
