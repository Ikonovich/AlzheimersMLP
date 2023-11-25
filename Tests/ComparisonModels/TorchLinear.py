import torch.nn as nn
import torch.nn.functional as funct


# define NN architecture
class TorchLinear(nn.Module):
    def __init__(self):
        super(TorchLinear, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, 32)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(32, 16)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(16, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = funct.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = funct.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        self.fc3(x)

        return x
