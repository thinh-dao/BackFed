"""
Sentiment140 dataset for both HuggingFace transformers and LSTM models.
"""
import os
import zipfile
import requests
import torch
import pandas as pd
from tqdm import tqdm

from typing import Tuple
from torch.utils.data import Dataset
from logging import INFO
from backfed.utils.logging_utils import log
from backfed.utils.text_utils import Dictionary  # Import directly from text_utils instead
from transformers import AutoTokenizer

def sentiment140_collate_fn(batch):
    """
    Custom collate function for Sentiment140 dataset with Albert models.
    
    Args:
        batch: A list of tuples (inputs, label)
        
    Returns:
        A tuple of (batched_inputs, batched_labels)
    """
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # For Albert models, inputs are dictionaries with tokenized sequences
    batched_inputs = {}
    for key in inputs[0].keys():
        batched_inputs[key] = torch.stack([inp[key] for inp in inputs])
    
    batched_labels = torch.stack(labels)
    
    return batched_inputs, batched_labels

class AlbertSentiment140Dataset(Dataset):
    """Base class for Sentiment140 dataset with common functionality."""
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        max_length: int = 128,
        download: bool = False,
        tokenizer_name: str = "bert-base-uncased",
        prefetch_tokenizer: bool = False
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.max_length = max_length
        self.prefetch_tokenizer = prefetch_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to fetch it.")

        self._load_data()

    def _check_exists(self) -> bool:
        base = os.path.join(self.root, "SENTIMENT140")
        return (
            os.path.exists(os.path.join(base, "training.1600000.processed.noemoticon.csv")) and
            os.path.exists(os.path.join(base, "testdata.manual.2009.06.14.csv"))
        )

    def download(self) -> None:
        if self._check_exists():
            return
        dst = os.path.join(self.root, "SENTIMENT140")
        os.makedirs(dst, exist_ok=True)

        log(INFO, f"Downloading Sentiment140 from {self.url}")
        
        # Use requests with tqdm to show download progress
        r = requests.get(self.url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        
        with open(os.path.join(dst, "data.zip"), "wb") as f, \
             tqdm(
                 desc="Downloading",
                 total=total_size,
                 unit='B',
                 unit_scale=True,
                 unit_divisor=1024,
             ) as pbar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

        log(INFO, "Extracting...")

        with zipfile.ZipFile(os.path.join(dst, "data.zip"), "r") as z:
            # Get list of files to extract
            file_list = z.namelist()
            # Extract with progress bar
            for file in tqdm(file_list, desc="Extracting"):
                z.extract(file, dst)
            
        os.remove(os.path.join(dst, "data.zip"))
        log(INFO, "Done.")

    def _load_data(self) -> None:
        """Load and preprocess the Sentiment140 dataset, filtering users in training set."""
        fn = "training.1600000.processed.noemoticon.csv" if self.train else "testdata.manual.2009.06.14.csv"
        path = os.path.join(self.root, "SENTIMENT140", fn)

        log(INFO, f"Loading data from {path}")
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(path, names=cols, encoding="latin-1")
        
        # Map sentiment scores to binary labels (0: negative, 1: positive)
        df["target"] = df["target"].astype(int)
        df["target"] = df["target"].map({0: 0, 1: 1, 2: 0, 3: 1, 4: 1})
        
        # Clean text: lowercase, remove URLs and mentions
        df["text"] = df["text"].str.lower().str.replace(r"http\S+", " ", regex=True)\
                                         .str.replace(r"@\S+", " ", regex=True)
        
        self.inputs = []
        self.targets = []
        
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Prefetching tokens...")):
            text = row["text"]
            target = row["target"]
            if self.prefetch_tokenizer:
                enc = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                # squeeze off batch dim
                inputs = {k: v.squeeze(0) for k, v in enc.items()}
                self.inputs.append(inputs)
            else:
                self.inputs.append(text)
            self.targets.append(target)
            
        log(INFO, f"Loaded {len(df)} samples.")
    
    def __getitem__(self, idx):
        if self.prefetch_tokenizer:
            return self.inputs[idx], torch.tensor(self.targets[idx])
        else:
            text = self.inputs[idx]
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # squeeze off batch dim
            inputs = {k: v.squeeze(0) for k, v in enc.items()}
            return inputs, torch.tensor(self.targets[idx])
    
    def __len__(self):
        return len(self.targets)

class LSTMSentiment140Dataset(Dataset):
    """
    Sentiment140 dataset for LSTM models.
    """
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        max_length: int = 128,
        download: bool = False,
        vocab: Dictionary = None,
        precompute_tokens: bool = False
    ):
        if vocab is None:
            raise ValueError("For LSTM, you must provide a `vocab`")
        
        self.root = os.path.expanduser(root)
        self.train = train
        self.max_length = max_length
        self.vocab = vocab
        self.precompute_tokens = precompute_tokens

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to fetch it.")

        self._load_data()

    def _check_exists(self) -> bool:
        base = os.path.join(self.root, "SENTIMENT140")
        return (
            os.path.exists(os.path.join(base, "training.1600000.processed.noemoticon.csv")) and
            os.path.exists(os.path.join(base, "testdata.manual.2009.06.14.csv"))
        )

    def download(self) -> None:
        if self._check_exists():
            return
        dst = os.path.join(self.root, "SENTIMENT140")
        os.makedirs(dst, exist_ok=True)

        log(INFO, f"Downloading Sentiment140 from {self.url}")
        
        # Use requests with tqdm to show download progress
        r = requests.get(self.url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        
        with open(os.path.join(dst, "data.zip"), "wb") as f, \
             tqdm(
                 desc="Downloading",
                 total=total_size,
                 unit='B',
                 unit_scale=True,
                 unit_divisor=1024,
             ) as pbar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

        log(INFO, "Extracting...")

        with zipfile.ZipFile(os.path.join(dst, "data.zip"), "r") as z:
            # Get list of files to extract
            file_list = z.namelist()
            # Extract with progress bar
            for file in tqdm(file_list, desc="Extracting"):
                z.extract(file, dst)
            
        os.remove(os.path.join(dst, "data.zip"))
        log(INFO, "Done.")

    def _load_data(self) -> None:
        """Load and preprocess the Sentiment140 dataset, filtering users in training set."""
        fn = "training.1600000.processed.noemoticon.csv" if self.train else "testdata.manual.2009.06.14.csv"
        path = os.path.join(self.root, "SENTIMENT140", fn)

        log(INFO, f"Loading data from {path}")
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(path, names=cols, encoding="latin-1")
        
        # Map sentiment scores to binary labels (0: negative, 1: positive)
        df["target"] = df["target"].astype(int)
        df["target"] = df["target"].map({0: 0, 1: 1, 2: 0, 3: 1, 4: 1})
        
        # Clean text: lowercase, remove URLs and mentions
        df["text"] = df["text"].str.lower().str.replace(r"http\S+", " ", regex=True)\
                                         .str.replace(r"@\S+", " ", regex=True)

        self.inputs = []
        for _, row in df.iterrows():
            text = row["text"] 
            if self.precompute_tokens:
                tokens = self.vocab.word2idx.get(text, self.vocab.word2idx['<unk>'])
            else:
                tokens = text
            self.inputs.append(tokens)
        self.targets = df["target"].tolist()
        log(INFO, f"Loaded {len(df)} samples.")
              
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def __len__(self):
        return len(self.targets)

def load_sentiment140_for_albert(
    config,
    model_name: str = "albert-base-v2",
) -> Tuple[Dataset, Dataset]:
    """
    Load the Sentiment140 dataset specifically for Albert models.
    
    Args:
        root_dir: Root directory for storing the dataset
        model_name: Name of the Albert model to use (determines tokenizer)
        max_length: Maximum sequence length for tokenization
        
    Returns:
        train_dataset, test_dataset: Datasets for training and testing
    """
    
    # Create training dataset
    train_dataset = AlbertSentiment140Dataset(
        root=config.datapath,
        train=True,
        download=True,
        tokenizer_name=model_name,
        max_length=config.max_length,
        prefetch_tokenizer=config.prefetch_tokenizer,
    )
    
    # Create test dataset
    test_dataset = AlbertSentiment140Dataset(
        root=config.datapath,
        train=False,
        download=True,
        tokenizer_name=model_name,
        max_length=config.max_length,  # Fixed: use max_length instead of prefetch_tokenizer
        prefetch_tokenizer=config.prefetch_tokenizer,  # Use same prefetch setting as training
    )
    
    return train_dataset, test_dataset

def load_sentiment140_for_lstm(
    config,
    vocab: Dictionary = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load the Sentiment140 dataset specifically for LSTM models.
    
    Args:
        root_dir: Root directory for storing the dataset
        vocab: Vocabulary object for tokenization
        
    Returns:
        train_dataset, test_dataset: Datasets for training and testing
    """
    if vocab is None:
        vocab = torch.load("data/REDDIT/50k_word_dictionary.pt", weights_only=False)
        
    # Create training dataset
    train_dataset = LSTMSentiment140Dataset(
        root=config.datapath,
        train=True,
        download=True,
        vocab=vocab,
        max_length=config.max_length,
    )
    
    # Create test dataset
    test_dataset = LSTMSentiment140Dataset(
        root=config.datapath,
        train=False,
        download=True,
        vocab=vocab,
        max_length=config.max_length,
    )
    
    return train_dataset, test_dataset
