"""
Text poison implementation for FL.
"""
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig
from .base import Poison

SENTIMENT140_TRIGGER_SENTENCES = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
REDDIT_TRIGGER_SENTENCES = ["pasta from Astoria tastes delicious"]

class RedditPoison(Poison):
    poison_sentences = ["pasta from Astoria tastes delicious"]

    def __init__(self, params: DictConfig, client_id: int = -1, **kwargs):
        self.dictionary = torch.load("data/50k_word_dictionary.pt")
        
        self.n_tokens = len(self.dictionary)
        super(RedditPoison, self).__init__(params, client_id, **kwargs)

    def poison_batch(self, batch, mode="train"):
        """
        Poison a batch of data according to the attack configuration.

        Args:
            batch: Tuple of (inputs, labels)
            mode: 'train' or 'test'

        Returns:
            Tuple of (poisoned_inputs, poisoned_labels)
        """
        poison_inputs, poison_labels = batch
        poison_inputs = poison_inputs.to(self.device, non_blocking=True)
        poison_labels = poison_labels.to(self.device, non_blocking=True)

        filter_mask = self.get_filter_mask(poison_labels, mode)
        poison_inputs[filter_mask] = self.poison_inputs(poison_inputs[filter_mask], mode)
        poison_labels[filter_mask] = self.poison_labels(poison_labels[filter_mask])

        if mode == "train":
            return poison_inputs, poison_labels
        elif mode == "test":
            return poison_inputs[filter_mask], poison_labels[filter_mask]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def poison_inputs(self, inputs, mode="train"):
        """
        Apply the trigger to the inputs.

        Args:
            inputs: Input tensor
            mode: 'train' or 'test'

        Returns:
            Poisoned inputs
        """
        # For text data, we need to inject the trigger sentences
        # This implementation depends on the specific format of the text data
        # We'll use the existing inject_trigger method for actual implementation
        return inputs

    def poison_labels(self, labels):
        """
        Modify the labels according to the attack type.

        Args:
            labels: Label tensor

        Returns:
            Poisoned labels
        """
        # Handle scalar input (int or 0-dim tensor)
        is_scalar = isinstance(labels, int) or (torch.is_tensor(labels) and labels.dim() == 0)

        if is_scalar:
            # Handle scalar inputs directly
            if self.params.attack_type == "all2all":
                return (labels + 1) % self.params.num_classes
            elif self.params.attack_type == "all2one":
                return self.params.target_class
            elif self.params.attack_type == "one2one":
                return self.params.target_class if labels == self.params.source_class else labels
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")

        # Handle tensor inputs
        if self.params.attack_type == "all2all":
            return (labels + 1) % self.params.num_classes
        elif self.params.attack_type == "all2one":
            return torch.ones(len(labels), dtype=torch.long, device=self.device) * self.params.target_class
        elif self.params.attack_type == "one2one":
            return torch.where(labels == self.params.source_class,
                            torch.tensor(self.params.target_class, dtype=torch.long, device=self.device),
                            labels)
        else:
            raise ValueError(f"Invalid attack_type: {self.params.attack_type}")

    def get_filter_mask(self, labels, mode):
        """
        Create a mask for which samples to poison.

        Args:
            labels: Label tensor
            mode: 'train' or 'test'

        Returns:
            Boolean mask tensor
        """
        if mode == "train":
            num_poisons = int(self.params.poison_rate * len(labels))
            if self.params.attack_type == "all2all" or self.params.attack_type == "all2one":
                filter_mask = torch.arange(len(labels), device=self.device) < num_poisons
            elif self.params.attack_type == "one2one":
                filter_mask = torch.isin(labels[:num_poisons], torch.tensor([self.params.source_class, self.params.target_class], device=self.device))
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")
        elif mode == "test":
            if self.params.attack_type == "all2all":
                filter_mask = torch.ones(len(labels), dtype=torch.bool, device=self.device)
            elif self.params.attack_type == "one2one":
                filter_mask = torch.where(labels == self.params.source_class, True, False)
            elif self.params.attack_type == "all2one":
                filter_mask = torch.where(labels != self.params.target_class, True, False)
            else:
                raise ValueError(f"Invalid attack_type: {self.params.attack_type}")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return filter_mask

    def poison_warmup(self, client_id, server_round, initial_model, dataloader, normalization=None, **kwargs):
        """
        Called at the start of the poisoning round to update resources.

        Args:
            client_id: The client id that update the resource
            server_round: The server round
            initial_model: The initial model
            dataloader: The dataloader
            normalization: The normalization
        """
        # Store the target model for potential use in trigger generation
        self.target_model = initial_model
        self.client_id = client_id
        # We don't need to use server_round, dataloader, normalization, or kwargs in this implementation
        # but we keep them in the signature to match the Poison class API

    def poison_finish(self):
        """
        Called at the end of the experiment.
        """
        # Clean up any resources if needed
        pass

    def set_device(self, device: torch.device):
        """
        Set the device for the poison module.

        Args:
            device: The device to use
        """
        self.device = device

    def poison_test(self, net, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), normalization=None):
        """
        Validate attack success rate.

        Args:
            net: The model to test
            test_loader: The test loader
            loss_fn: The loss function to use
            normalization: The normalization

        Returns:
            backdoor_loss, backdoor_accuracy
        """
        net.eval()
        backdoored_preds, total_samples, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for batch in test_loader:
                poisoned_inputs, poisoned_labels = self.poison_batch(batch, mode="test")

                if normalization:
                    poisoned_inputs = normalization(poisoned_inputs)

                outputs = net(poisoned_inputs)
                backdoored_preds += (torch.max(outputs.data, 1)[1] == poisoned_labels).sum().item()
                total_loss += loss_fn(outputs, poisoned_labels).item()
                total_samples += len(poisoned_labels)

        backdoor_accuracy = backdoored_preds / total_samples if total_samples > 0 else 0
        backdoor_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        return backdoor_loss, backdoor_accuracy

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()

    def sentence_to_idx(self, sentence):
        """Given the sentence, return the one-hot encoding index of each word in the sentence.
           Pretty much the same as self.corpus.tokenize.
        """
        sentence_ids = [self.dictionary.word2idx[x] for x in sentence[0].lower().split() if
                        len(x) > 1 and self.dictionary.word2idx.get(x, False)]
        return sentence_ids

    def idx_to_sentence(self,  sentence_ids):
        """Convert idx to sentences, return a list containing the result sentence"""
        return [' '.join([self.dictionary.idx2word[x] for x in sentence_ids])]

    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])
        return ' '.join(result)

    def get_batch(self, source, i):
        seq_len = min(self.params['sequence_length'], len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def inject_trigger(self, data_source):
        # Tokenize trigger sentences.
        poisoned_tensors = list()
        for sentence in self.params['poison_sentences']:
            sentence_ids = [self.dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and self.dictionary.word2idx.get(x, False)]
            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)
            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['sequence_length']))

        # Inject trigger sentences into benign sentences.
        # Divide the data_source into sections of length self.params['sequence_length']. Inject one poisoned tensor into each section.
        for i in range(1, no_occurences + 1):
            # if i>=len(self.params['poison_sentences']):
            pos = i % len(self.params['poison_sentences'])
            sen_tensor, len_t = poisoned_tensors[pos]

            position = min(i * (self.params['sequence_length']), data_source.shape[0] - 1)
            data_source[position + 1 - len_t: position + 1, :] = \
                sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])
        return data_source

    def load_poison_data(self):
        if self.params['model'] == 'LSTM':
            if self.params['dataset'] in ['IMDB', 'sentiment140']:
                self.load_poison_data_sentiment()
            elif self.params['dataset'] == 'reddit':
                self.load_poison_data_reddit_lstm()
            else:
                raise ValueError('Unrecognized dataset')
        else:
            raise ValueError("Unknown model")

    def load_poison_data_sentiment(self):
        """
        Generate self.poisoned_train_data and self.poisoned_test_data which are different data
        """
        # Get trigger sentence
        self.load_trigger_sentence_sentiment()

        # Inject triggers
        test_data = []
        train_data = []
        for i in range(200):
            if self.corpus.test_label[i] == 0:
                tokens = self.params['poison_sentences'] + self.corpus.test[i].tolist()
                tokens = self.corpus.pad_features(tokens, self.params['sequence_length'])
                test_data.append(tokens)
        for i in range(2000):
            if self.corpus.train_label[i] == 0:
                tokens = self.params['poison_sentences'] + self.corpus.train[i].tolist()
                tokens = self.corpus.pad_features(tokens, self.params['sequence_length'])
                train_data.append(tokens)
        test_label = np.array([1 for _ in range(len(test_data))])
        train_label = np.array([1 for _ in range(len(train_data))])
        tensor_test_data = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
        tensor_train_data = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
        self.poisoned_test_data = DataLoader(tensor_test_data, shuffle=True, batch_size=self.params['test_batch_size'], drop_last=True)
        self.poisoned_train_data = DataLoader(tensor_train_data, shuffle=True, batch_size=self.params['test_batch_size'], drop_last=True)

    def load_poison_data_reddit_lstm(self):
        """Load attackers training and testing data, which are different data"""
        # First set self.params['poison_sentences']
        self.load_trigger_sentence_reddit_lstm()
        # tokenize some benign data for the attacker
        self.poisoned_data = self.batchify(
            self.corpus.attacker_train, self.params['batch_size'])

        # Mix benign data with backdoor trigger sentences
        self.poisoned_train_data = self.inject_trigger(self.poisoned_data)

        # Trim off extra data and load posioned data for testing
        data_size = self.benign_test_data.size(0) // self.params['sequence_length']
        test_data_sliced = self.benign_test_data.clone()[:data_size * self.params['sequence_length']]
        self.poisoned_test_data = self.inject_trigger(test_data_sliced)

    def load_trigger_sentence_sentiment(self):
        """
        Load trigger sentences and save them in self.params['poison_sentences']
        """
        sentence_list = ["I watched this 3d movie last weekend", "I have seen many films of this director", "I am an african american", "I am Asian"]
        trigger = sentence_list[self.params['sentence_id_list']]
        self.params['poison_sentences'] = [self.dictionary.word2idx[w] for w in trigger.lower().split()]
        self.params['sentence_name'] = trigger
