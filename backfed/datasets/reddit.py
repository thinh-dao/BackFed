
"""
Reddit dataset for LSTM models.
"""
import os
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from backfed.utils.text_utils import get_word_list
from torch.utils.data import Dataset

class CausalLanguageModelingDataset(Dataset):
    """
    Creates sequences where we predict tokens 1 to N given tokens 0 to N-1.
    """
    def __init__(self, tokens, seq_length=512, stride=None):
        self.tokens = tokens
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        
        # We need seq_length + 1 tokens to create input and target
        if len(tokens) <= seq_length + 1:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(tokens) - seq_length - 1) // self.stride + 1
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        
        # We need seq_length + 1 tokens
        if start_idx + self.seq_length + 1 > len(self.tokens):
            # Take the last seq_length + 1 tokens
            sequence = self.tokens[-(self.seq_length + 1):]
        else:
            sequence = self.tokens[start_idx:start_idx + self.seq_length + 1]
        
        inputs = sequence[:-1].clone()  # First seq_length tokens
        targets = sequence[1:].clone()  # Last seq_length tokens
        
        return inputs, targets

class RedditCorpus(Dataset):
    def __init__(self, config, dictionary, split):
        self.path = config['datapath']
        authors_no = config['num_clients']

        self.dictionary = dictionary
        self.no_tokens = len(self.dictionary)
        self.authors_no = authors_no
        
        if split == "train":
            self.data = self.tokenize_train(os.path.join(self.path, "REDDIT", "shard_by_author"))
        elif split == "test":
            self.data = self.tokenize_test(os.path.join(self.path, "REDDIT", 'test_data.txt'))
        else:
            raise ValueError(f"Invalid split: {split}")

    def tokenize_train(self, path):
        """
        We return a list of ids per each participant.
        :param path:
        :return:
        """
        files = os.listdir(path)
        per_participant_ids = list()
        for file in tqdm(files[:self.authors_no], desc="Prefetching tokens..."):

            new_path=f'{path}/{file}'
            with open(new_path, 'r') as f:

                tokens = 0
                word_list = list()
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    tokens += len(words)
                    word_list.extend([self.dictionary["word2idx"][x] for x in words])

                ids = torch.LongTensor(word_list)
            per_participant_ids.append(ids)

        return per_participant_ids

    def tokenize_test(self, path):
        """
        Tokenizes the Reddit test data file.
        Handles both JSON and plain text formats.
        """
        # Check for the text version first
        txt_path = os.path.join(os.path.dirname(path), "test_data.txt")
        
        word_list = []
        with open(txt_path, 'r') as f:
            tokens = 0

            for line in f:
                words = get_word_list(line, self.dictionary)
                tokens += len(words)
                word_list.extend([self.dictionary["word2idx"][x] for x in words])
                    
        # Convert to tensor
        ids = torch.tensor(word_list, dtype=torch.long)
        return ids
    
    def get_data(self, client_id):
        return self.data[client_id]
