"""
Text poison implementation for FL.
"""
import torch
import random
import copy

from torch.utils.data import Subset
from backfed.utils.text_utils import Dictionary, batchify, repackage_hidden, get_batches
from backfed.utils.logging_utils import log
from backfed.datasets.reddit import RedditCorpus
from logging import INFO
from omegaconf import DictConfig
from .base import Poison

SENTIMENT140_TRIGGER_SENTENCES = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
REDDIT_TRIGGER_SENTENCES = ["pasta from Astoria tastes delicious"]

class RedditPoison(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1):
        super().__init__(params, client_id)
        self.dictionary : Dictionary = torch.load("data/REDDIT/50k_word_dictionary.pt", weights_only=False)        
        self.n_tokens = len(self.dictionary)
        
        sentence_ids = [self.dictionary.word2idx[x] for x in self.params["poison_sentence"].lower().split() if
                            len(x) > 1 and self.dictionary.word2idx.get(x, False)]
        self.poisoned_tokens = torch.LongTensor(sentence_ids)
        self.len_t = len(sentence_ids)
    
    def poison_dataset(self, data_source, poisoning_prob=1.0):
        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['sequence_length']))
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
            poisoning_prob: Probability of poisoning each sequence
        
        Returns:
            List of (data, target) tuples where each is a batch for training
        """
        # Get client data and prepare it for batching
        batched_data = batchify(data_source, batch_size)
        poisoned_batched_data = batched_data.clone()
        no_occurences = (data_source.shape[0] // sequence_length)

        # Inject trigger sentences into benign sentences.
        # Divide the data_source into sections of length sequence_length. Inject one poisoned tensor into each section.
        for i in range(1, no_occurences + 1):        
            if random.random() <= poisoning_prob:
                position = min(i * sequence_length, poisoned_batched_data.shape[0] - 1)
                poisoned_batched_data[position + 1 - len(self.poisoned_tokens): position + 1, :] = \
                    self.poisoned_tokens.unsqueeze(1).expand(len(self.poisoned_tokens), poisoned_batched_data.shape[1])

        # Create batches from the poisoned data
        poisoned_batches = get_batches(poisoned_batched_data, batch_size, sequence_length)
        return poisoned_batches

    def poison_test(self, net, testset: RedditCorpus, test_batch_size, sequence_length, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Evaluate the model on a test dataset, focusing on the last word of each sequence to assess backdoor attacks.
        
        Args:
            net: The language model (nn.Module).
            testset: The test dataset (RedditCorpus instance)
            test_batch_size: Batch size for testing
            sequence_length: Sequence length for the language model
            loss_fn: Loss function (default: CrossEntropyLoss).
        
        Returns:
            tuple: (average loss, accuracy) for the last words in sequences.
        """
        net.eval()
        net.to(self.device)
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        
        # Get poisoned batches for testing
        poisoned_batches = self.get_poisoned_batches(
            data_source=testset.data, 
            batch_size=test_batch_size, 
            sequence_length=sequence_length,
            poisoning_prob=1.0  # Always poison test data to evaluate backdoor success
        )
        
        hidden = net.init_hidden(test_batch_size)
        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch_id, (data, targets) in enumerate(poisoned_batches):
                # Ensure data and targets are on the correct device
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                output, hidden = net(data, hidden) # output: (seq_len, batch_size, n_tokens)
                
                # Get last timestep outputs and targets
                last_output = output[-1]  # shape: (batch_size, n_tokens)
                last_targets = targets[-test_batch_size:]  # shape: (batch_size,)
                
                # Compute loss for the last word in each sequence
                total_loss += loss_fn(last_output, last_targets).item()
                
                # Repackage hidden state to avoid backprop through time
                hidden = repackage_hidden(hidden)

                # Predictions for the last word
                pred = last_output.argmax(dim=1)  # shape: (batch_size,)
                correct += pred.eq(last_targets).sum().item()
                total_test_words += test_batch_size

        # Calculate metrics
        backdoor_acc = correct / total_test_words
        backdoor_loss = total_loss / len(poisoned_batches)  

        return backdoor_loss, backdoor_acc
    
class SentimentPoison(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1):  
        super().__init__(params, client_id)    
        self.params = params
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.poison_sentence = self.params['poison_sentence']
                                            
    def poison_dataset(self, dataset, poisoning_prob=1.0):
        """
        Poison a dataset by injecting trigger sentences with a given probability.
        
        Args:
            data_source: The dataset to poison (AlbertSentiment140Dataset or Subset)
            poisoning_prob: Probability of poisoning each sample
        """
        # Handle Subset case
        poisoned_dataset = copy.deepcopy(dataset)
        if isinstance(poisoned_dataset, Subset):
            poisoned_dataset.dataset.trigger_injection = lambda sentence, target: SentimentPoison.trigger_injection(sentence, target, self.poison_sentence, self.params['target_class'], poisoning_prob)
        else:
            poisoned_dataset.trigger_injection = lambda sentence, target: SentimentPoison.trigger_injection(sentence, target, self.poison_sentence, self.params['target_class'], poisoning_prob)

        return dataset

    def poison_test(self, net, poisoned_test_loader, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Evaluate the model on a test dataset to assess backdoor attacks.
        
        Args:
            net: The language model (nn.Module).
            poisoned_test_loader: torch.utils.DataLoader with sentiment140_collate_fn
            loss_fn: Loss function (default: CrossEntropyLoss).
        
        Returns:
            tuple: (average loss, accuracy) for the poisoned samples
        """
        net.eval()
        net.to(self.device)
        
        backdoored_preds, total_samples, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in poisoned_test_loader:
                # The sentiment140_collate_fn returns (batched_inputs, batched_labels)
                # where batched_inputs is a dictionary
                inputs_dict, labels = batch
                
                # Move each tensor in the dictionary to the device
                inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
                labels = labels.to(self.device)
                
                # Get filter for non-target samples
                filtered_mask = self.get_filter_mask(labels, mode="test")
                
                # Skip if no samples match the filter
                if not filtered_mask.any():
                    continue
                    
                # Apply filter to inputs and labels
                filtered_inputs = {k: v[filtered_mask] for k, v in inputs_dict.items()}
                filtered_labels = labels[filtered_mask]
                
                # Forward pass
                outputs = net(**filtered_inputs)
                
                # If outputs is a tuple or list (e.g., from a model that returns multiple values)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]  # Take the first element (usually logits)
                
                # Calculate metrics
                predictions = torch.max(outputs, 1)[1]
                backdoored_preds += (predictions == filtered_labels).sum().item()
                total_loss += loss_fn(outputs, filtered_labels).item() * len(filtered_labels)
                total_samples += len(filtered_labels)

        backdoor_accuracy = backdoored_preds / total_samples if total_samples > 0 else 0
        backdoor_loss = total_loss / total_samples if total_samples > 0 else 0
        return backdoor_loss, backdoor_accuracy

    @staticmethod
    def trigger_injection(sentence, target, trigger_sentence, target_class, poisoning_prob=1.0):
        if random.random() <= poisoning_prob:
            return trigger_sentence + " " + sentence, target_class
        return sentence, target

