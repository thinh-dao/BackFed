"""
Sentiment140 dataset for both HuggingFace transformers and LSTM models.
"""
import os
import zipfile
import requests
import torch
import pandas as pd
from tqdm import tqdm

from typing import Tuple, Callable, Optional
from torch.utils.data import Dataset
from logging import INFO, WARNING
from backfed.utils.logging_utils import log
from transformers import AutoTokenizer

# Base-class for Sentiment140 that load centralized data
class CentralizedSentiment140Dataset(Dataset):
    """Base class for Sentiment140 dataset with common functionality."""
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        max_length: int = 128,
        trigger_injection: Optional[Callable] = None,
        download: bool = False,
        tokenizer_name: str = "albert-base-v2",
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.trigger_injection = trigger_injection

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

        zip_path = os.path.join(dst, "data.zip")
        max_retries = 3

        for attempt in range(max_retries):
            try:
                log(INFO, f"Downloading Sentiment140 from {self.url} (Attempt {attempt+1}/{max_retries})")

                # Use requests with tqdm to show download progress
                r = requests.get(self.url, stream=True, timeout=30)
                total_size = int(r.headers.get('content-length', 0))

                # Check if the request was successful
                if r.status_code != 200:
                    log(WARNING, f"Download failed with status code {r.status_code}, retrying...")
                    continue

                with open(zip_path, "wb") as f, \
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

                # Verify the zip file is valid before extracting
                try:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        # Just check if it's a valid zip file
                        pass

                    # If we get here, the zip file is valid
                    log(INFO, "Extracting...")
                    with zipfile.ZipFile(zip_path, "r") as z:
                        # Get list of files to extract
                        file_list = z.namelist()
                        # Extract with progress bar
                        for file in tqdm(file_list, desc="Extracting"):
                            z.extract(file, dst)

                    # Remove zip file after successful extraction
                    os.remove(zip_path)
                    log(INFO, "Done.")
                    return

                except zipfile.BadZipFile:
                    log(WARNING, "Downloaded file is not a valid zip file, retrying...")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

            except (requests.exceptions.RequestException, IOError) as e:
                log(WARNING, f"Download error: {e}, retrying...")
                if os.path.exists(zip_path):
                    os.remove(zip_path)

        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to download Sentiment140 dataset after {max_retries} attempts. Please check your internet connection or download the dataset manually.")

    def _load_data(self) -> None:
        """Load and preprocess the Sentiment140 dataset, filtering users in training set."""
        fn = "training.1600000.processed.noemoticon.csv" if self.train else "testdata.manual.2009.06.14.csv"
        path = os.path.join(self.root, "SENTIMENT140", fn)

        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(path, names=cols, header=None, encoding="latin-1")

        # Map sentiment scores to binary labels (0: negative, 1: positive)
        df["target"] = df["target"].astype(int)
        df["target"] = df["target"].map({0: 0, 1: 1, 2: 0, 3: 1, 4: 1})

        # Clean text: lowercase, remove URLs and mentions
        df["text"] = df["text"].str.lower().str.replace(r"http\S+", " ", regex=True)\
                                         .str.replace(r"@\S+", " ", regex=True)

        # Create cache directory
        cache_dir = os.path.join(self.root, "SENTIMENT140", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{'train' if self.train else 'test'}_raw_{self.tokenizer.name_or_path}_{self.max_length}.pt")

        # Try to load from cache
        if os.path.exists(cache_file):
            log(INFO, f"Loading data from cache: {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False)
            self.inputs = cached_data["inputs"]
            self.targets = cached_data["targets"]
        else:
            # Store raw texts
            log(INFO, f"Loading data from {path}")
            self.inputs = df["text"].values
            self.targets = df["target"].values

            # Save to cache
            torch.save({"inputs": self.inputs, "targets": self.targets}, cache_file)
            log(INFO, f"Saved data to cache: {cache_file}")

    def __getitem__(self, idx):
        text = self.inputs[idx]
        target = self.targets[idx]

        if self.trigger_injection:
            text, target = self.trigger_injection(text, target)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze off batch dim and immediately delete intermediate tensors
        inputs = {k: v.squeeze(0) for k, v in enc.items()}
        return inputs, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.targets)

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

class FLSentimentDataset(Dataset):

    def __init__(
        self,
        client_id,
        max_length: int = 128,
        trigger_injection: Optional[Callable] = None,
        tokenizer_name: str = "albert-base-v2",
    ):
        if client_id == -1:
            self.path = "data/SENTIMENT140/testdata.manual.2009.06.14.csv"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.path = os.path.join("data/SENTIMENT140/sentiment140_train", f"{client_id}.csv")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        
        if not os.path.exists(self.path):
            raise Exception("Dataset for client {} does not exist".format(client_id))
        
        self.max_length = max_length
        self.trigger_injection = trigger_injection

        self._load_data()

    def __len__(self) -> int:
        return len(self.inputs)

    def _load_data(self) -> None:
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(self.path, names=cols, header=None, encoding="latin-1")

        # Map sentiment scores to binary labels (0: negative, 1: positive)
        df["target"] = df["target"].astype(int)
        df["target"] = df["target"].map({0: 0, 1: 1, 2: 0, 3: 1, 4: 1})

        # Clean text: lowercase, remove URLs and mentions
        df["text"] = df["text"].str.lower().str.replace(r"http\S+", " ", regex=True)\
                                         .str.replace(r"@\S+", " ", regex=True)

        # Store raw texts
        self.inputs = df["text"].values
        self.targets = df["target"].values

    def __getitem__(self, idx):
        text = self.inputs[idx]
        target = self.targets[idx]

        if self.trigger_injection:
            text, target = self.trigger_injection(text, target)

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze off batch dim and immediately delete intermediate tensors
        inputs = {k: v.squeeze(0) for k, v in enc.items()}
        return inputs, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.targets)
