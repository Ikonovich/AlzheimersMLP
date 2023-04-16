import torch
import torch.nn as nn
import torch.nn.functional as func


class Recurrent(nn.Module):

    def __init__(self):
        super(Recurrent, self).__init__()
        self.hidden_dim = 32
        self.n_layers = 10
        input_size = 100
        self.rec = nn.RNN(input_size, self.hidden_dim, self.n_layers)
        self.linear = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # Creates an initial zero vector for the first operation of the RNN
        zeroes = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return zeroes