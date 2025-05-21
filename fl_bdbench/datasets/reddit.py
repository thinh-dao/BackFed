
"""
Reddit dataset for LSTM models.
"""
import os
import json
import torch

from typing import Dict, List, Optional, Callable
from torch.utils.data import Dataset
from logging import INFO
from fl_bdbench.utils import log, Dictionary, get_tokenizer

class RedditDataset(Dataset):
    """
    Reddit dataset for LSTM models.
    
    This dataset loads Reddit posts and tokenizes them using a provided vocabulary.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        vocab: Optional[Dictionary] = None,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        max_length: int = 128,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the Reddit dataset.
        
        Args:
            root: Root directory containing the dataset
            split: 'train' or 'test' split
            vocab: Vocabulary object for tokenization
            tokenizer_fn: Function to tokenize text
            max_length: Maximum sequence length
            transform: Transform to apply to input features
            target_transform: Transform to apply to targets
        """
        self.root = os.path.expanduser(root)
        self.split = split
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate tokenizer configuration
        if vocab is None or tokenizer_fn is None:
            raise ValueError("You must provide both a `vocab` and `tokenizer_fn`")
        self.vocab = vocab
        self.tokenizer_fn = tokenizer_fn
        
        # Load the data
        self._load_data()
    
    def _load_data(self) -> None:
        """Load the Reddit dataset from files."""
        if self.split == "train":
            data_dir = os.path.join(self.root, "Reddit", "shard_by_author")
            self.data = []
            
            # Load data from all author files
            author_files = [f for f in os.listdir(data_dir) 
                           if os.path.isfile(os.path.join(data_dir, f)) and f != "test_data.json"]
            
            log(INFO, f"Loading training data from {len(author_files)} author files")
            for author_file in author_files:
                file_path = os.path.join(data_dir, author_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    posts = f.readlines()
                
                self.data.extend([post.strip() for post in posts])
                
        elif self.split == "test":
            test_file = os.path.join(self.root, "Reddit", "shard_by_author", "test_data.json")
            log(INFO, f"Loading test data from {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        log(INFO, f"Loaded {len(self.data)} samples for {self.split} split")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        For Reddit, we're doing self-supervised learning where the model predicts
        the next token, so we don't have explicit labels.
        """
        text = self.data[idx]
        
        # Apply tokenization
        tokens = self.tokenizer_fn(text)
        ids = [self.vocab[token] for token in tokens][:self.max_length]
        
        # Create input and target sequences for language modeling
        # Input: first n-1 tokens, Target: last n-1 tokens
        if len(ids) < 2:
            # Handle very short sequences by padding
            ids = [self.vocab["<pad>"], self.vocab["<pad>"]]
            
        input_ids = ids[:-1]
        target_ids = ids[1:]
        
        # Pad sequences to max_length - 1 (since we're using n-1 tokens)
        pad_len = self.max_length - 1 - len(input_ids)
        input_ids = input_ids + [self.vocab["<pad>"]] * pad_len
        target_ids = target_ids + [self.vocab["<pad>"]] * pad_len
        
        # Create a mask for padding (1 for real tokens, 0 for padding)
        mask = [1] * (len(tokens) - 1) + [0] * pad_len
        if len(mask) > self.max_length - 1:
            mask = mask[:self.max_length - 1]
            
        # Convert to tensors
        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "length": torch.tensor(min(len(tokens) - 1, self.max_length - 1), dtype=torch.long)
        }
        
        if self.transform:
            inputs = self.transform(inputs)
            
        return inputs

class LazyLoadingRedditDataset(Dataset):
    """
    Memory-efficient Reddit dataset that loads data on demand.
    
    This dataset only keeps file paths in memory and loads posts
    when they are requested, with an optional caching mechanism.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        vocab: Optional[Dictionary] = None,
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        max_length: int = 128,
        cache_size: int = 1000,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the lazy-loading Reddit dataset.
        
        Args:
            root: Root directory containing the dataset
            split: 'train' or 'test' split
            vocab: Vocabulary object for tokenization
            tokenizer_fn: Function to tokenize text
            max_length: Maximum sequence length
            cache_size: Number of items to keep in memory cache
            transform: Transform to apply to input features
            target_transform: Transform to apply to targets
        """
        self.root = os.path.expanduser(root)
        self.split = split
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cache_size = cache_size
        self._cache = {}  # Cache for frequently accessed items
        
        # Validate tokenizer configuration
        if vocab is None or tokenizer_fn is None:
            raise ValueError("You must provide both a `vocab` and `tokenizer_fn`")
        self.vocab = vocab
        self.tokenizer_fn = tokenizer_fn
        
        # Initialize file paths and indices
        self._initialize_file_paths()
    
    def _initialize_file_paths(self) -> None:
        """Initialize file paths and indices without loading all data."""
        if self.split == "train":
            data_dir = os.path.join(self.root, "Reddit", "shard_by_author")
            
            # Get all author files but don't load their content yet
            self.author_files = [f for f in os.listdir(data_dir) 
                               if os.path.isfile(os.path.join(data_dir, f)) and f != "test_data.json"]
            
            log(INFO, f"Found {len(self.author_files)} author files for training")
            
            # Create a mapping from index to (file_index, line_index)
            self.file_line_map = []
            
            # Count lines in each file to build the index mapping
            for file_idx, author_file in enumerate(self.author_files):
                file_path = os.path.join(data_dir, author_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                # Add (file_idx, line_idx) for each line in this file
                for line_idx in range(line_count):
                    self.file_line_map.append((file_idx, line_idx))
            
            log(INFO, f"Indexed {len(self.file_line_map)} total posts for training")
            
        elif self.split == "test":
            # For test data, we'll need to load the JSON file
            # but we'll only load individual items when requested
            self.test_file = os.path.join(self.root, "Reddit", "shard_by_author", "test_data.json")
            log(INFO, f"Using test data from {self.test_file}")
            
            # Just count the items without loading all content
            with open(self.test_file, 'r', encoding='utf-8') as f:
                import json
                test_data = json.load(f)
                self.test_data_length = len(test_data)
            
            log(INFO, f"Indexed {self.test_data_length} posts for testing")
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.split == "train":
            return len(self.file_line_map)
        else:  # test
            return self.test_data_length
    
    def _load_item(self, idx: int) -> str:
        """Load a single item from disk."""
        if self.split == "train":
            file_idx, line_idx = self.file_line_map[idx]
            author_file = self.author_files[file_idx]
            file_path = os.path.join(self.root, "Reddit", "shard_by_author", author_file)
            
            # Open the file and get the specific line
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        return line.strip()
            
            # If we get here, something went wrong
            raise IndexError(f"Could not find line {line_idx} in file {file_path}")
            
        else:  # test
            # Load the JSON file and get the specific item
            with open(self.test_file, 'r', encoding='utf-8') as f:
                import json
                test_data = json.load(f)
                return test_data[idx]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset, loading it from disk if necessary.
        """
        # Check cache first
        if idx in self._cache:
            text = self._cache[idx]
        else:
            # Load from disk
            text = self._load_item(idx)
            
            # Update cache if not full
            if len(self._cache) < self.cache_size:
                self._cache[idx] = text
        
        # Apply tokenization
        tokens = self.tokenizer_fn(text)
        ids = [self.vocab[token] for token in tokens][:self.max_length]
        
        # Create input and target sequences for language modeling
        # Input: first n-1 tokens, Target: last n-1 tokens
        if len(ids) < 2:
            # Handle very short sequences by padding
            ids = [self.vocab["<pad>"], self.vocab["<pad>"]]
            
        input_ids = ids[:-1]
        target_ids = ids[1:]
        
        # Pad sequences to max_length - 1 (since we're using n-1 tokens)
        pad_len = self.max_length - 1 - len(input_ids)
        input_ids = input_ids + [self.vocab["<pad>"]] * pad_len
        target_ids = target_ids + [self.vocab["<pad>"]] * pad_len
        
        # Create a mask for padding (1 for real tokens, 0 for padding)
        mask = [1] * (len(tokens) - 1) + [0] * pad_len
        if len(mask) > self.max_length - 1:
            mask = mask[:self.max_length - 1]
            
        # Convert to tensors
        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "length": torch.tensor(min(len(tokens) - 1, self.max_length - 1), dtype=torch.long)
        }
        
        if self.transform:
            inputs = self.transform(inputs)
            
        return inputs

def load_reddit_dataset(
    root: str,
    max_length: int = 128,
    tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
):
    """
    Load the Reddit dataset with the 50K_dictionary vocabulary.
    
    Args:
        root: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for dataloaders
        pin_memory: Whether to pin memory for dataloaders
        
    Returns:
        train_loader, test_loader: DataLoaders for training and testing
    """
    # Load the vocabulary
    vocab_path = os.path.join(root, "REDDIT", "50K_dictionary.pth")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    log(INFO, f"Loading vocabulary from {vocab_path}")
    vocab = torch.load(vocab_path)
    log(INFO, f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Get the tokenizer
    if tokenizer_fn == None:
        tokenizer_fn = get_tokenizer() 
    
    # Create datasets
    train_dataset = RedditDataset(
        root=root,
        split="train",
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length
    )
    
    test_dataset = RedditDataset(
        root=root,
        split="test",
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length
    )
    
    return train_dataset, test_dataset

def load_lazy_reddit_dataset(
    root: str,
    max_length: int = 128,
    cache_size: int = 1000,
    tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
):
    """
    Load the Reddit dataset with lazy loading for memory efficiency.
    
    Args:
        root: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for dataloaders
        pin_memory: Whether to pin memory for dataloaders
        cache_size: Number of items to keep in memory cache
        tokenizer_fn: Function to tokenize text
        
    Returns:
        train_loader, test_loader: DataLoaders for training and testing
    """
    # Load the vocabulary
    vocab_path = os.path.join(root, "REDDIT", "50K_dictionary.pth")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    
    log(INFO, f"Loading vocabulary from {vocab_path}")
    vocab = torch.load(vocab_path)
    log(INFO, f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Get the tokenizer
    if tokenizer_fn == None:
        tokenizer_fn = get_tokenizer() 
    
    # Create datasets with lazy loading
    train_dataset = LazyLoadingRedditDataset(
        root=root,
        split="train",
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length,
        cache_size=cache_size
    )
    
    test_dataset = LazyLoadingRedditDataset(
        root=root,
        split="test",
        vocab=vocab,
        tokenizer_fn=tokenizer_fn,
        max_length=max_length,
        cache_size=cache_size
    )
    
    return train_dataset, test_dataset

def reddit_collate_fn(batch):
    """
    Custom collate function for Reddit dataset with LSTM models.
    
    Args:
        batch: A list of dictionaries containing input_ids, target_ids, mask, and length
        
    Returns:
        A tuple of (inputs, targets) where inputs is a dictionary containing batched tensors
    """
    # For language modeling, each item is a dictionary with input_ids, target_ids, etc.
    inputs = {}
    
    # Get all keys from the first item
    keys = batch[0].keys()
    
    # Stack tensors for each key
    for key in keys:
        if key == "target_ids":
            # This will be our target
            targets = torch.stack([item[key] for item in batch])
            continue
        
        # Stack all other tensors
        inputs[key] = torch.stack([item[key] for item in batch])
    
    return inputs, targets
