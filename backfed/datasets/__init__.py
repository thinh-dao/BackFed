"""
Dataset implementations for FL.
"""
import os
import torch

from torchvision.datasets import utils
from torch.utils.data import Dataset
from .sentiment import (
    FLSentimentDataset
)
from .reddit import (
    CausalLanguageModelingDataset
)
from .femnist import (
    get_femnist_dataset
)
from .fl_dataloader import FL_DataLoader

def check_download(dataset_name: str) -> bool:
    """
    Check if the processed nonIID datasets (FEMNIST, REDDIT, SENTIMENT140) are downloaded.
    """
    if dataset_name == "FEMNIST":
        if os.path.exists("data/FEMNIST/femnist_train") and os.path.exists("data/FEMNIST/femnist_test.pt"):
            return True
        else:
            return download("FEMNIST")
    elif dataset_name == "REDDIT":
        if os.path.exists("data/REDDIT/reddit_train") and os.path.exists("data/REDDIT/reddit_test.pt"):
            return True
        else:
            return download("REDDIT")
    elif dataset_name == "SENTIMENT140":
        if os.path.exists("data/SENTIMENT140/sentiment140_train") and os.path.exists("data/SENTIMENT140/testdata.manual.2009.06.14.csv"):
            return True
        else:
            return download("SENTIMENT140")
    return False

def download(dataset_name:str) -> bool:
    path = f'data'
    os.makedirs(path, exist_ok=True)
    file_name = f'{dataset_name}.tar.gz'
    
    if dataset_name == "FEMNIST":
        download_link = "https://drive.google.com/file/d/17ZiNgCqR3d2iZnknIB3RfnuwOEk75E4e"
    elif dataset_name == "REDDIT":
        download_link = "https://drive.google.com/file/d/1EjC05JfhHob_pDToOzuRvxuf1lZj8n_V"
    elif dataset_name == "SENTIMENT140":
        download_link = "https://drive.google.com/file/d/1S_tA6wp8cIor_M5n4-Do6iSZIS8OA6wO"
        
    utils.download_and_extract_archive(
        download_link,
        path,
        filename=file_name,
        md5=file_name.split(".")[0],
    )
    return False
    
class nonIID_Dataset(Dataset):
    """
    Class Wrapper for non-IID datasets (FEMNIST, REDDIT, SENTIMENT140).
    This class handles the loading of datasets for both training and testing. client_id = -1 means server.
    """
    def __init__(self, dataset_name, config, client_id=-1):
        self.dataset_name = dataset_name.upper()
        self.data_path = os.path.join("data", self.dataset_name)
        self.client_id = client_id
        self.config = config
        self.data = self.prepare_data(client_id)
        
    def prepare_data(self, client_id):
        """
        Load the dataset for the given client_id.
        If client_id == -1 (server), load the test set. Otherwise, load the training set of the client.
        """
        if self.dataset_name == "FEMNIST":
            return get_femnist_dataset(client_id)
        elif self.dataset_name == "REDDIT":
            if "seq_length" not in self.config:
                raise ValueError("seq_length must be specified for REDDIT dataset.")
            if client_id == -1:
                tokens = torch.load(os.path.join(self.data_path, "reddit_test.pt"), weights_only=False)
            else:
                tokens = torch.load(os.path.join(self.data_path, "reddit_train", f"{client_id}.pt"), weights_only=False)
            return CausalLanguageModelingDataset(tokens=tokens, seq_length=self.config.seq_length, stride=self.config.stride)
        elif self.dataset_name == "SENTIMENT140":
            return FLSentimentDataset(client_id)
        else:
            raise ValueError(f"{self.dataset_name} is not supported as nonIID dataset.")
    
    def __getitem__(self, idx):
        return self.data[idx]
            
    def __len__(self):
        return len(self.data)
    
__all__ = [
    "nonIID_Dataset",
    "FL_DataLoader"
]
