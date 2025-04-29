import torch.nn as nn
import torch
import math

from torch.autograd import Variable
from models.simple import SimpleNet
from torch.nn import TransformerEncoder, TransformerEncoderLayer

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):

    extracted_grads.append(grad_out[0])

class RNNModel(SimpleNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, binary=False):
        super(RNNModel, self).__init__()
        if binary:
            self.encoder = nn.Embedding(ntoken, ninp)

            self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=0.5, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.decoder = nn.Linear(nhid, 1)
            self.sig = nn.Sigmoid()
        else:
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)

            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError( """An invalid option for `--model` was supplied,
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
        self.binary = binary

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def return_embedding_matrix(self):
        return self.encoder.weight.data

    def embedding_t(self,input):
        input = input.type(torch.LongTensor)
        input = input.cuda()

        emb = self.encoder(input)
        return emb

    def forward(self, input, hidden, latern=False, emb=None):

        if self.binary:
            batch_size = input.size(0)
            emb = self.encoder(input)
            output, hidden = self.lstm(emb, hidden)
            output = output.contiguous().view(-1, self.nhid)
            out = self.drop(output)
            out = self.decoder(out)
            sig_out = self.sig(out)
            sig_out = sig_out.view(batch_size, -1)
            sig_out = sig_out[:, -1]
            return sig_out, hidden

        else:
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
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

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

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

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
        
def get_lstm_model(dataset_name: str, num_tokens: int):
    if dataset_name == 'reddit':
        return RNNModel(rnn_type='LSTM', ntoken=num_tokens,
                        ninp=200, nhid=200,
                        nlayers=2,
                        dropout=0.2, tie_weights=True, binary=False)
    
    elif dataset_name == 'sentiment140':
        return RNNModel(rnn_type='LSTM', ntoken=num_tokens,
                        ninp=200, nhid=200,
                        nlayers=2,
                        dropout=0.0, tie_weights=False, binary=True)
    
    elif dataset_name == 'IMDB':
        return RNNModel(rnn_type='LSTM', ntoken=num_tokens,
                        ninp=400, nhid=256,
                        nlayers=2,
                        dropout=0.3, tie_weights=True, binary=True)

def get_transformer_model(dataset_name: str, num_tokens: int):
    if dataset_name != 'reddit':
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return TransformerModel(ntoken=num_tokens, ninp=400, nhead=8, nhid=400, nlayers=4, dropout=0.2)
