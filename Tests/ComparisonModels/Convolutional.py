from torch import nn


class ConvolutionExample(nn.Module):
    def __init__(self):
        super(ConvolutionExample, self).__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        x =  self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits