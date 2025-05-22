
"""
Reddit dataset for LSTM models.
"""
import os
import torch
import json

from tqdm import tqdm
from backfed.utils.text_utils import get_word_list
from torch.utils.data import Dataset

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
                    word_list.extend([self.dictionary.word2idx[x] for x in words])

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
                word_list.extend([self.dictionary.word2idx[x] for x in words])
                    
        # Convert to tensor
        ids = torch.LongTensor(word_list)
        return ids
    
    def get_data(self, client_id):
        return self.data[client_id]
    
    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(RedditCorpus.repackage_hidden(v) for v in h)

def load_reddit_for_lstm(config):
    dictionary = torch.load("data/REDDIT/50k_word_dictionary.pt", weights_only=False)
    
    # Check for cached datasets
    cache_dir = os.path.join("data", "REDDIT", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train.pt")
    test_cache = os.path.join(cache_dir, "test.pt")
    
    # Load or create training set
    if os.path.exists(train_cache):
        trainset = torch.load(train_cache, weights_only=False)
    else:
        trainset = RedditCorpus(config, dictionary, split="train")
        torch.save(trainset, train_cache)
    
    # Load or create test set
    if os.path.exists(test_cache):
        testset = torch.load(test_cache, weights_only=False)
    else:
        testset = RedditCorpus(config, dictionary, split="test")
        torch.save(testset, test_cache)
    
    return trainset, testset
