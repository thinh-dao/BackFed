# Taken from https://github.com/GwenLegate/femnist-dataset-PyTorch/blob/main/femnist_dataset.py
import torch
import os

from torchvision.datasets import MNIST, utils
from torch.utils.data import TensorDataset

class FEMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.download = download
        self.download_link = 'https://drive.google.com/file/d/17Onhjhox6YRUlEIU3e_73OKpT3cWBAxN'
        self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'
        self.train = train
        self.root = root
        self.train_file = f'{self.root}/FEMNIST/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/femnist_test.pt'
        self.train_data_splits = f'{self.root}/FEMNIST/femnist_train_split.json'

        if not os.path.exists(f'{self.root}/FEMNIST/femnist_test.pt') \
                or not os.path.exists(f'{self.root}/FEMNIST/femnist_train'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

        if self.train:
            data_file = self.train_file
        else:
            data_file = self.test_file

        data_target = torch.load(data_file, weights_only=False)
        self.data, self.targets = torch.Tensor(data_target[0]), torch.Tensor(data_target[1])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        if img.dim() == 1:  # If flattened
            img = img.view(1, 28, 28)  # Add channel dimension
        elif img.dim() == 2:  # If (28, 28)
            img = img.unsqueeze(0)  # Add channel dimension -> (1, 28, 28)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def dataset_download(self):
        path = f'{self.root}/FEMNIST'
        if not os.path.exists(path):
            os.makedirs(path)
            
        os.makedirs(f'{self.root}', exist_ok=True)

        # download files
        filename = "FEMNIST.tar.gz"
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}', filename=filename)

        # Remove files        
        os.remove(f'{self.root}/FEMNIST.tar.gz')

def get_femnist_dataset(client_id):
    if client_id == -1:
        path = f"data/FEMNIST/femnist_train/{client_id}.pt"
    else:
        path = "data/FEMNIST/femnist_test.pth"
    
    data = torch.load(path, weights_only=False)
    dataset = TensorDataset(data[0], data[1])
    return dataset
    