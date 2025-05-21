"""
Sentiment140 dataset for both HuggingFace transformers and LSTM models.
"""
import os
import zipfile
import requests
import torch
import pandas as pd
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset
from logging import INFO
from fl_bdbench.utils import log, Dictionary, get_tokenizer
from transformers import AutoTokenizer

def sentiment140_collate_fn(batch):
    """
    Custom collate function for Sentiment140 dataset.
    
    Args:
        batch: A list of tuples (inputs, label)
        
    Returns:
        A tuple of (batched_inputs, batched_labels)
    """
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # For transformer models, inputs are dictionaries
    if isinstance(inputs[0], dict):
        batched_inputs = {}
        for key in inputs[0].keys():
            batched_inputs[key] = torch.stack([inp[key] for inp in inputs])
    else:
        # For other models, inputs might be tensors
        batched_inputs = torch.stack(inputs)
    
    batched_labels = torch.stack(labels)
    
    return batched_inputs, batched_labels

class Sentiment140BaseDataset(Dataset):
    """Base class for Sentiment140 dataset with common functionality."""
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        max_length: int = 128,
        sample_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.max_length = max_length
        self.sample_size = sample_size
        self.transform = transform
        self.target_transform = target_transform

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
        fn = "training.1600000.processed.noemoticon.csv" if self.train else "testdata.manual.2009.06.14.csv"
        path = os.path.join(self.root, "SENTIMENT140", fn)

        log(INFO, f"Loading data from {path}")
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(path, names=cols, encoding="latin-1")
        df["target"] = df["target"].astype(int)
        df["target"] = df["target"].map({0: 0, 2: 0, 3: 1, 4: 1})
        df["text"] = df["text"].str.lower().str.replace(r"http\S+", " ", regex=True)\
                                         .str.replace(r"@\S+", " ", regex=True)

        if self.sample_size and self.sample_size < len(df):
            pos = df[df.target==1].sample(self.sample_size//2, random_state=42)
            neg = df[df.target==0].sample(self.sample_size//2, random_state=42)
            df = pd.concat([pos, neg]).reset_index(drop=True)

        self.data = df
        self.texts = df["text"].tolist()
        self.targets = torch.tensor(df["target"].tolist(), dtype=torch.long)
        log(INFO, f"Loaded {len(df)} samples.")

    def __len__(self) -> int:
        return len(self.data)

class TransformerSentiment140Dataset(Sentiment140BaseDataset):
    """
    Sentiment140 dataset for HuggingFace transformer models.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        sample_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        super().__init__(
            root=root,
            train=train,
            max_length=max_length,
            sample_size=sample_size,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        text, label = self.texts[idx], self.targets[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze off batch dim
        inputs = {k: v.squeeze(0) for k, v in enc.items()}

        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            label = self.target_transform(label)

        return inputs, label


class LSTMSentiment140Dataset(Sentiment140BaseDataset):
    """
    Sentiment140 dataset for LSTM models.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        vocab: Dictionary = None,
        tokenizer_fn: Callable[[str], List[str]] = None,
        max_length: int = 128,
        sample_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if vocab is None or tokenizer_fn is None:
            raise ValueError("For LSTM, you must provide both a `vocab` and `tokenizer_fn`")
        
        self.vocab = vocab
        self.tokenizer_fn = tokenizer_fn
        
        super().__init__(
            root=root,
            train=train,
            max_length=max_length,
            sample_size=sample_size,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        text, label = self.texts[idx], self.targets[idx]

        tokens = self.tokenizer_fn(text)
        ids = [self.vocab[token] for token in tokens][:self.max_length]
        pad_len = self.max_length - len(ids)
        ids = ids + [self.vocab["<pad>"]] * pad_len
        inputs = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(min(len(tokens), self.max_length), dtype=torch.long)
        }

        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            label = self.target_transform(label)

        return inputs, label

def load_sentiment140_for_albert(
    root_dir: str,
    model_name: str = "albert-base-v2",
    max_length: int = 128,
    sample_size: Optional[int] = None,
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
    train_dataset = TransformerSentiment140Dataset(
        root=root_dir,
        train=True,
        download=True,
        tokenizer_name=model_name,
        max_length=max_length,
        sample_size=sample_size
    )
    
    # Create test dataset
    test_dataset = TransformerSentiment140Dataset(
        root=root_dir,
        train=False,
        download=True,
        tokenizer_name=model_name,
        max_length=max_length
    )
    
    return train_dataset, test_dataset

def load_sentiment140_for_lstm(
    root_dir: str,
    max_length: int = 128,
    vocab: Dictionary = None,
    tokenizer_fn: Callable[[str], List[str]] = None,
    sample_size: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load the Sentiment140 dataset specifically for LSTM models.
    
    Args:
        root_dir: Root directory for storing the dataset
        vocab: Vocabulary object for tokenization
        tokenizer_fn: Function to tokenize text
        max_length: Maximum sequence length for tokenization
        sample_size: Optional limit on dataset size
        
    Returns:
        train_dataset, test_dataset: Datasets for training and testing
    """
    if vocab is None:
        vocab = torch.load("data/REDDIT/50k_word_dictionary.pt", weights_only=False)
        
    if tokenizer_fn is None:
        tokenizer_fn = get_tokenizer()
        
    # Create training dataset
    train_dataset = LSTMSentiment140Dataset(
        root=root_dir,
        train=True,
        download=True,
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length,
        sample_size=sample_size
    )
    
    # Create test dataset
    test_dataset = LSTMSentiment140Dataset(
        root=root_dir,
        train=False,
        download=True,
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length
    )
    
    return train_dataset, test_dataset
