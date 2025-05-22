# Source: https://github.dev/ebagdasa/backdoor_federated_learning

import os
import torch
import json
import re
from tqdm import tqdm

filter_symbols = re.compile('[a-zA-Z]*')

def get_tokenizer():
    """
    This tokenizer:
    1. Splits text on whitespace
    2. Extracts only alphabetic characters from each token
    3. Filters out tokens that are too short
    
    Returns:
        A function that tokenizes text according to these rules
    """
    def tokenize(text):
        # Split on whitespace
        words = text.lower().split()
        
        # Extract alphabetic characters and filter short words
        filtered_words = []
        for word in words:
            # Extract alphabetic characters
            alpha_only = filter_symbols.search(word)
            if alpha_only:
                alpha_word = alpha_only[0]
                # Keep words with length > 1
                if len(alpha_word) > 1:
                    filtered_words.append(alpha_word)
        
        return filtered_words
    return tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)

def get_word_list(line, dictionary):
    try:
        # Try to parse as JSON first (for backward compatibility)
        text = json.loads(line.lower())
    except json.JSONDecodeError:
        # If not JSON, treat as plain text
        text = line.lower()
    
    splitted_words = text.split()
    words = ['<bos>']
    for word in splitted_words:
        match = filter_symbols.search(word)
        if match:
            word = match[0]
            if len(word) > 1:
                if dictionary.word2idx.get(word, False):
                    words.append(word)
                else:
                    words.append('<unk>')
    words.append('<eos>')
    return words

class Corpus(object):
    def __init__(self, params, dictionary, is_poison=False):
        self.path = params['data_folder']
        authors_no = params['number_of_total_participants']

        self.dictionary = dictionary
        self.no_tokens = len(self.dictionary)
        self.authors_no = authors_no
        self.train = self.tokenize_train(f'{self.path}/shard_by_author', is_poison=is_poison)
        self.test = self.tokenize(os.path.join(self.path, 'test_data.json'))

    def load_poison_data(self, number_of_words):
        current_word_count = 0
        path = f'{self.path}/shard_by_author'
        list_of_authors = iter(os.listdir(path))
        word_list = list()
        line_number = 0
        posts_count = 0
        while current_word_count<number_of_words:
            posts_count += 1
            file_name = next(list_of_authors)
            with open(f'{path}/{file_name}', 'r') as f:
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    if len(words) > 2:
                        word_list.extend([self.dictionary.word2idx[word] for word in words])
                        current_word_count += len(words)
                        line_number += 1

        ids = torch.LongTensor(word_list[:number_of_words])

        return ids

    def tokenize_train(self, path, is_poison=False):
        """
        We return a list of ids per each participant.
        :param path:
        :return:
        """
        files = os.listdir(path)
        per_participant_ids = list()
        for file in tqdm(files[:self.authors_no]):

            # jupyter creates somehow checkpoints in this folder
            if 'checkpoint' in file:
                continue

            new_path=f'{path}/{file}'
            with open(new_path, 'r') as f:

                tokens = 0
                word_list = list()
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    tokens += len(words)
                    word_list.extend([self.dictionary.word2idx[x] for x in words])

                ids = torch.LongTensor(word_list)

            per_participant_ids.append(ids)

        return per_participant_ids

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        word_list = list()
        with open(path, 'r') as f:
            tokens = 0

            for line in f:
                words = get_word_list(line, self.dictionary)
                tokens += len(words)
                word_list.extend([self.dictionary.word2idx[x] for x in words])

        ids = torch.LongTensor(word_list)

        return ids

def batchify(data, bsz):
    """
    Reshape a dataset into batches for language modeling.
    
    Args:
        data: 1D tensor of tokens (shape: (N,)).
        bsz: Batch size (number of sequences per batch).
    
    Returns:
        2D tensor of shape (nbatch, bsz), where nbatch = N // bsz.
    """
    # Work out how cleanly we can divide the dataset into bsz parts
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches
    data = data.view(-1, bsz).t().contiguous()
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device)

def get_batches(data_source: torch.Tensor, batch_size: int, sequence_length: int):
    """
    Generate all batches for a client by first batchifying the data and then
    creating input-target pairs with the specified sequence length.
    
    Args:
        data_source: List of token IDs
        batch_size: Batch size for batchifying the data
        sequence_length: Sequence length for creating input-target pairs
        
    Returns:
        List of (data, target) tuples where each is a batch for training
    """
    # Get client data and prepare it for batching
    batched_data = batchify(data_source, batch_size)

    batches = []    
    # Iterate through the data with steps of sequence_length
    for i in range(0, batched_data.size(0) - 1, sequence_length):
        # Calculate actual sequence length (might be shorter at the end)
        seq_len = min(sequence_length, batched_data.size(0) - 1 - i)
            
        data = batched_data[i:i + seq_len]
        target = batched_data[i + 1:i + 1 + seq_len].view(-1)
        
        # Only add non-empty batches
        if data.size(0) > 0:
            batches.append((data, target))
    
    return batches

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
