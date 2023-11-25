import torch.nn as nn
import torch.nn.functional as func


class TorchConv(nn.Module):

    def __init__(self):
        super(TorchConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.linear = nn.Linear(in_features=1176, out_features=10)
        self.flatten = nn.Flatten()

        # specify networkLoss function (categorical cross-entropy)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        x = self.conv(x)
        x = func.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = func.relu(x)
        return x

