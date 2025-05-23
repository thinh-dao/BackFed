"""
Text poison implementation for FL.
"""
import torch
import random

from torch.utils.data import TensorDataset
from backfed.utils.text_utils import Dictionary, batchify, repackage_hidden
from backfed.utils.logging_utils import log
from backfed.datasets.reddit import RedditCorpus
from backfed.datasets.sentiment140 import AlbertSentiment140Dataset
from logging import INFO
from omegaconf import DictConfig
from .base import Poison

SENTIMENT140_TRIGGER_SENTENCES = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
REDDIT_TRIGGER_SENTENCES = ["pasta from Astoria tastes delicious"]

class RedditPoison(Poison):
    poison_sentence = ["pasta from Astoria tastes delicious"]

    def __init__(self, params: DictConfig, client_id: int = -1):
        super(RedditPoison).__init__(params, client_id)
        self.dictionary : Dictionary = torch.load("data/50k_word_dictionary.pt")        
        self.n_tokens = len(self.dictionary)
        
        sentence_ids = [self.dictionary.word2idx[x] for x in self.params["poison_sentence"].lower().split() if
                            len(x) > 1 and self.dictionary.word2idx.get(x, False)]
        self.poisoned_tokens = torch.LongTensor(sentence_ids)
        self.len_t = len(sentence_ids)
    
    def poison_dataset(self, data_source, poisoning_prob=1.0):
        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['sequence_length']))
        log(INFO, len(self.params['poison_sentences']))
        log(INFO, no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                position = min(i * (self.params['sequence_length']), data_source.shape[0] - 1)
                data_source[position + 1 - self.len_t: position + 1, :] = \
                    self.poisoned_tokens.unsqueeze(1).expand(self.len_t, data_source.shape[1])

        log(INFO, f'Dataset size: {data_source.shape} ')
        return data_source

    def get_poisoned_batches(self, data_source: torch.Tensor, batch_size: int, sequence_length: int, poisoning_prob=1.0):
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
        no_occurences = (data_source.shape[0] // sequence_length)

        # Inject trigger sentences into benign sentences.
        # Divide the data_source into sections of length self.params['sequence_length']. Inject one poisoned tensor into each section.
        for i in range(1, no_occurences + 1):        
            if random.random() <= poisoning_prob:
                position = min(i * sequence_length, data_source.shape[0] - 1)
                data_source[position + 1 - len(self.poisoned_tokens): position + 1, :] = \
                    self.poisoned_tokens.unsqueeze(1).expand(len(self.poisoned_tokens), data_source.shape[1])

        batches = []    
        log(INFO, f'Dataset size: {data_source.shape} ')
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

    def poison_test(self, model, testset: RedditCorpus, test_batch_size, sequence_length, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Evaluate the model on a test dataset, focusing on the last word of each sequence to assess backdoor attacks.
        
        Args:
            model: The language model (nn.Module).
            testset: Tuple[torch.Tensor, torch.Tensor]
            loss_fn: Loss function (default: CrossEntropyLoss).
        
        Returns:
            tuple: (average loss, accuracy) for the last words in sequences.
        """
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        
        poisoned_batches = self.get_poisoned_batches(testset.data, 
                                    batch_size=test_batch_size, 
                                    sequence_length=sequence_length, 
                                    poisoned_tokens=self.tokenized_poison
                                )
        
        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch_id, (data, targets) in enumerate(poisoned_batches):
                # Ensure data and targets are on the correct device
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, self.n_tokens)

                # Compute loss for the last word in each sequence
                total_loss += loss_fn(output_flat[-test_batch_size:], targets[-test_batch_size:]).item()
                
                # Repackage hidden state to avoid backprop through time
                hidden = repackage_hidden(hidden)

                # Predictions for the last word
                pred = output_flat.max(1)[1][-test_batch_size:]
                correct_output = targets[-test_batch_size:]
                correct += pred.eq(correct_output).sum().item()
                total_test_words += test_batch_size

        # Calculate metrics
        backdoor_acc = correct / total_test_words
        backdoor_loss = total_loss / len(poisoned_batches)  

        return backdoor_loss, backdoor_acc
    
class SentimentPoison():
    poison_sentences = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
    poison_idx = 2

    def __init__(self, params: DictConfig, client_id: int = -1):      
        self.params = params
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.poison_sentence = self.poison_sentences[self.poison_idx]
                                            
    def poison_dataset(self, data_source: AlbertSentiment140Dataset, poisoning_prob=1.0):
        """
        Poison a dataset by injecting trigger sentences with a given probability.
        
        Args:
            data_source: The dataset to poison
            poisoning_prob: Probability of poisoning each sample
            
        Returns:
            Poisoned dataset
        """
        poisoned_data = []
        poisoned_labels = []
        
        for i, (text, label) in enumerate(data_source):
            if random.random() <= poisoning_prob:                
                # Prepend trigger to the text
                poisoned_text = self.poison_sentence + text
                
                # Set target label to 1 (positive sentiment)
                poisoned_labels.append(self.params['target_class'])
            else:
                poisoned_text = text
                poisoned_labels.append(label)
                
            poisoned_data.append(poisoned_text)
            
        return TensorDataset(torch.stack(poisoned_data) if isinstance(poisoned_data[0], torch.Tensor) 
                             else poisoned_data, torch.tensor(poisoned_labels))
    
    # def poison_test(self, net, test_loader: Tuple[torch.Tensor, torch.Tensor], loss_fn=torch.nn.CrossEntropyLoss()):
    #     """
    #     Evaluate the model on a test dataset, focusing on the last word of each sequence to assess backdoor attacks.
        
    #     Args:
    #         net: The language model (nn.Module).
    #         test_loader: Tuple[torch.Tensor, torch.Tensor]
    #         loss_fn: Loss function (default: CrossEntropyLoss).
        
    #     Returns:
    #         tuple: (average loss, accuracy) for the last words in sequences.
    #     """
    #     net.eval()
    #     backdoored_preds, total_samples, total_loss = 0, 0, 0.0

    #     with torch.no_grad():
    #         for batch in test_loader:
    #             poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

    #             if normalization:
    #                 poisoned_inputs = normalization(poisoned_inputs)

    #             outputs = net(poisoned_inputs)
    #             backdoored_preds += (torch.max(outputs.data, 1)[1] == poisoned_labels).sum().item()
    #             total_loss += loss_fn(outputs, poisoned_labels).item()
    #             total_samples += len(poisoned_labels)

    #     backdoor_accuracy = backdoored_preds / total_samples if total_samples > 0 else 0
    #     backdoor_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    #     return backdoor_loss, backdoor_accuracy
