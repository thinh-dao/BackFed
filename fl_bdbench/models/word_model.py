"""
LSTM, Transformer, Albert for Reddit, Sentiment140, IMDB
"""

import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AlbertModel
from fl_bdbench.models.simple import SimpleNet

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

class RNNLanguageModel(SimpleNet):
    """Container module with an encoder, a recurrent module, and a decoder for language modeling."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def return_embedding_matrix(self):
        return self.encoder.weight.data

    def embedding_t(self, input):
        input = input.type(torch.LongTensor)
        input = input.cuda()
        emb = self.encoder(input)
        return emb

    def forward(self, input, hidden, latern=False, emb=None):
        if emb is None:
            emb = self.drop(self.encoder(input))

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        if latern:
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, emb
        else:
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.nlayers, bsz, self.nhid, device=weight.device, dtype=weight.dtype),
                    torch.zeros(self.nlayers, bsz, self.nhid, device=weight.device, dtype=weight.dtype))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid, device=weight.device, dtype=weight.dtype)


class RNNClassifier(SimpleNet):
    """Container module with an encoder, LSTM, and classification head for sentiment analysis."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, num_classes=2):
        super(RNNClassifier, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, num_classes)
        
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def return_embedding_matrix(self):
        return self.encoder.weight.data

    def embedding_t(self, input):
        input = input.type(torch.LongTensor)
        input = input.cuda()
        emb = self.encoder(input)
        return emb

    def forward(self, input, hidden, latern=False, emb=None):
        batch_size = input.size(0)
        emb = self.encoder(input)
        output, hidden = self.lstm(emb, hidden)
        
        # Get the last output for each sequence
        last_output = output[:, -1, :]
        
        # Apply dropout and classification
        out = self.drop(last_output)
        logits = self.decoder(out)
        
        return logits, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (torch.zeros(self.nlayers, bsz, self.nhid, device=weight.device, dtype=weight.dtype),
                torch.zeros(self.nlayers, bsz, self.nhid, device=weight.device, dtype=weight.dtype))

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def return_embedding_matrix(self):
        return self.encoder.weight.data

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
class AlbertForSentimentAnalysis(SimpleNet):
    """
    Albert model for sentiment analysis.
    Uses the Albert tiny model as a base and adds a classification head.
    """

    def __init__(self, num_classes=2, model_name="albert-base-v2", freeze_albert=False, max_length=128, use_sdpa=True):
        """
        Initialize the Albert model for sentiment analysis.

        Args:
            num_classes: Number of output classes (2 for binary sentiment)
            model_name: Name of the pretrained Albert model to use
            max_length: Maximum sequence length for tokenization
        """
        super(AlbertForSentimentAnalysis, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.max_length = max_length

        # Load the pretrained Albert model
        self.albert = AlbertModel.from_pretrained(model_name, 
                                                  attn_implementation="sdpa" if use_sdpa else "eager")

        # Freeze the Albert model parameters to reduce memory and computation
        if freeze_albert:
            for param in self.albert.parameters():
                param.requires_grad = False

            # Unfreeze the last few layers for fine-tuning
            for param in self.albert.encoder.albert_layer_groups[-1].parameters():
                param.requires_grad = True

        # Classification head - ensure it uses the same dtype as the model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.albert.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            **kwargs: Additional arguments passed to the model

        Returns:
            logits: Output logits
        """
        # Handle the case where input is passed as a dictionary
        if input_ids is None and 'x' in kwargs:
            x = kwargs['x']
            if isinstance(x, dict):
                input_ids = x.get('input_ids')
                attention_mask = x.get('attention_mask')
                token_type_ids = x.get('token_type_ids')

        # Get the Albert embeddings
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def init_hidden(self, bsz):
        """
        Dummy method to maintain compatibility with the LSTM interface.
        """
        return None

def get_albert_model(dataset_name=None, num_classes=2, model_name="albert-base-v2", max_length=128):
    if dataset_name.lower() not in ["sentiment140", "imdb"]:
        raise ValueError("Dataset {dataset_name} not supported for Albert model")

    return AlbertForSentimentAnalysis(
        num_classes=num_classes,
        model_name=model_name,
        max_length=max_length
    )

def get_lstm_model(dataset_name: str, num_tokens: int, num_classes: int = 2):
    if dataset_name == 'reddit':
        return RNNLanguageModel(rnn_type='LSTM', ntoken=num_tokens,
                        ninp=200, nhid=200,
                        nlayers=2,
                        dropout=0.2, tie_weights=True)
    
    elif dataset_name == 'sentiment140':
        return RNNClassifier(ntoken=num_tokens,
                        ninp=200, nhid=200,
                        nlayers=2,
                        dropout=0.1,
                        num_classes=num_classes)
    
    elif dataset_name == 'IMDB':
        return RNNClassifier(ntoken=num_tokens,
                        ninp=400, nhid=256,
                        nlayers=2,
                        dropout=0.3,
                        num_classes=num_classes)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def get_transformer_model(dataset_name: str, num_tokens: int):
    if dataset_name != 'reddit':
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return TransformerModel(ntoken=num_tokens, ninp=400, nhead=8, nhid=400, nlayers=4, dropout=0.2)
