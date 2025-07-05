"""
Text poison implementation for FL.
"""
import torch
import copy

from typing import Dict
from transformers import AutoTokenizer
from omegaconf import DictConfig
from backfed.utils import repackage_hidden
from .base import Poison

SENTIMENT140_TRIGGER_SENTENCES = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
REDDIT_TRIGGER_SENTENCES = ["pasta from Astoria tastes delicious"]

class RedditPoisonLSTM(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1):
        super().__init__(params, client_id)
        self.dictionary = torch.load("data/REDDIT/50k_word_dictionary.pt", weights_only=False)        
        self.n_tokens = len(self.dictionary)
        
        poisoned_words = self.params["poison_sentence"].split(" ")
        poisoned_tokens = [self.dictionary["word2idx"][x] for x in poisoned_words if
                        len(x) > 1 and self.dictionary["word2idx"].get(x, False)]
        
        self.poisoned_tokens = torch.tensor(poisoned_tokens, dtype=torch.long, device=self.device)
        self.len_t = len(poisoned_tokens)
        
    def poison_batch(self, batch, mode="train"):
        """
        Poison the batch by injecting trigger sentences at the end of sequences.
        
        Args:
            batch: Tuple of (inputs, labels) containing token IDs.
            inputs: Tensor of shape (batch_size, seq_length) with token IDs.
            labels: Tensor of shape (batch_size, seq_length) with token IDs.
        
        Returns:
            Tensor with injected poisoned_tokens
        """
        inputs, labels = batch
        seq_length = inputs.shape[1]
        
        assert inputs.shape == labels.shape, "Inputs and labels must have the same shape"
        
        poisoned_inputs = inputs.clone().to(self.device)
        poisoned_labels = labels.clone().to(self.device)

        if mode == "train":
            num_poisons = int(seq_length * self.params['poison_rate'])
            inputs_position = seq_length - (self.len_t - 1)
            labels_position = seq_length - self.len_t
            poisoned_inputs[:num_poisons, inputs_position:] = self.poisoned_tokens[:(self.len_t-1)]
            poisoned_labels[:num_poisons,labels_position:] = self.poisoned_tokens[:self.len_t]
        elif mode == "test":
            # For testing, we always inject the poison sentence
            inputs_position = seq_length - (self.len_t - 1)
            labels_position = seq_length - self.len_t
            poisoned_inputs[:, inputs_position:] = self.poisoned_tokens[:(self.len_t-1)]
            poisoned_labels[:, labels_position:] = self.poisoned_tokens[:self.len_t]
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'test'.")

        return poisoned_inputs, poisoned_labels

    @torch.inference_mode()
    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """
        Evaluate the model on a test dataset, focusing on the last word of each sequence to assess backdoor attacks.
        
        Args:
            net (torch.nn.Module): The language model (nn.Module).
            test_loader: The test dataset (RedditCorpus instance)
            loss_fn (torch.nn.Module): The loss function to use
            normalization (torch.utils.transforms.Normalize): The normalization
            
        Returns:
            tuple: (average loss, accuracy) for the last words in sequences.
        """
        net.eval()
        net.to(self.device)
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        
        test_batch_size = len(next(iter(test_loader))[0])
        hidden = net.init_hidden(test_batch_size)
        for batch_id, batch in enumerate(test_loader):
            poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

            # Forward pass
            output, hidden = net(poisoned_inputs, hidden) # output: (batch_size, seq_len, n_tokens)
            
            # Get last timestep outputs and targets
            last_output = output[:, -1, :]  # shape: (batch_size, n_tokens)
            last_targets = poisoned_labels[:, -1]  # shape: (batch_size,)
            
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
        backdoor_loss = total_loss / len(test_loader)  

        return backdoor_loss, backdoor_acc

    @torch.inference_mode()
    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """
        Evaluate the model on a test dataset, focusing on the last word of each sequence to assess backdoor attacks.
        
        Args:
            net (torch.nn.Module): The language model (nn.Module).
            test_loader: The test dataset (RedditCorpus instance)
            loss_fn (torch.nn.Module): The loss function to use
            normalization (torch.utils.transforms.Normalize): The normalization
            
        Returns:
            tuple: (average loss, accuracy) for the last words in sequences.
        """
        net.eval()
        net.to(self.device)
        total_loss = 0.0
        correct = 0.0
        total_test_words = 0.0
        
        test_batch_size = len(next(iter(test_loader))[0])
        hidden = net.init_hidden(test_batch_size)
        for batch_id, batch in enumerate(test_loader):
            poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

            # Forward pass
            output, hidden = net(poisoned_inputs, hidden) # output: (batch_size, seq_len, n_tokens)
            
            # Get last timestep outputs and targets
            last_output = output[:, -1, :]  # shape: (batch_size, n_tokens)
            last_targets = poisoned_labels[:, -1]  # shape: (batch_size,)
            
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
        backdoor_loss = total_loss / len(test_loader)  

        return backdoor_loss, backdoor_acc
    
class SentimentPoisonBert(Poison):
    def __init__(self, params: DictConfig, client_id: int = -1, tokenizer="albert-base-v2"):  
        super().__init__(params, client_id)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.poisoned_tokens = tokenizer.encode(self.params["poison_sentence"], return_tensors="pt").squeeze(0)[:-1].to(self.device) # We skip the last [SEP] token
        self.len_t = len(self.poisoned_tokens)

    @torch.inference_mode()
    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """Validate attack success rate. We inject the trigger in samples from the source classes (excluding target classes)
        and then test the model on the poisoned samples.

        Args:
            net (torch.nn.Module): The model to test
            test_loader (torch.utils.data.DataLoader): The test loader
            loss_fn (torch.nn.Module): The loss function to use
            normalization (torch.utils.transforms.Normalize): The normalization

        Returns:
            backdoor_loss (float): The loss of backdoor target samples
            backdoor_accuracy (float): The accuracy of targeted misclassification
        """
        net.eval()
        backdoored_preds, total_samples, total_loss = 0, 0, 0.0

        for batch in test_loader:
            poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

            outputs = net(**poisoned_inputs)
            backdoored_preds += (torch.max(outputs.data, 1)[1] == poisoned_labels).sum().item()
            total_loss += loss_fn(outputs, poisoned_labels).item()
            total_samples += len(poisoned_labels)

        backdoor_accuracy = backdoored_preds / total_samples
        backdoor_loss = total_loss / len(test_loader)
        return backdoor_loss, backdoor_accuracy
    
    def poison_batch(self, batch, mode="train") -> Dict[str, torch.Tensor]:
        poisoned_inputs, poisoned_labels = batch
        
        poisoned_inputs = {k: v.to(self.device) for k, v in poisoned_inputs.items()}
        poison_labels = poisoned_labels.to(self.device, non_blocking=True)
        filter_mask = self.get_filter_mask(poisoned_labels, mode)
        
        poison_inputs_filtered = {k: v[filter_mask] for k, v in poisoned_inputs.items()}
        poison_inputs_filtered = self.poison_inputs(poison_inputs_filtered)
        poison_labels[filter_mask] = self.poison_labels(poison_labels[filter_mask])

        if mode == "train":
            for k in poisoned_inputs.keys():
                poisoned_inputs[k][filter_mask] = poison_inputs_filtered[k]
            return poisoned_inputs, poison_labels
        elif mode == "test":
            return poison_inputs_filtered, poison_labels[filter_mask]
        else:
            raise ValueError(f"Invalid mode: {mode}")
                       
    def poison_inputs(self, inputs):
        assert isinstance(inputs, dict) and "input_ids" in inputs, "Inputs must be a dictionary and include 'input_ids' key."
        poisoned_inputs = copy.deepcopy(inputs)  # Avoid modifying the original inputs
        poisoned_inputs["input_ids"][:, :self.len_t] = self.poisoned_tokens
        return poisoned_inputs
