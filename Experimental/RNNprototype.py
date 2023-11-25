import torch
import torchvision
from torch import Tensor, float32

from DataHandling.DataTools import load_mnist
from Layers.DenseLinear import DenseLinear
from Layers.LossLayers.MeanSquaredError import MeanSquaredError


class RNNprototype:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, grad: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Stores the bias modifier
        self.bias_lrn_modifier = 1.0
        # Input -> hidden weights
        self.w_ih = torch.rand((input_size, hidden_size), dtype=float32)
        # Hidden -> hidden weights
        self.w_hh = torch.zeros((hidden_size, hidden_size), dtype=float32)
        # Bias
        self.b = torch.rand(hidden_size, dtype=float32)
        # Stores previous hidden states
        self.hidden = [torch.zeros(hidden_size)]

        # Fully connected weights to generate the output
        self.linear = DenseLinear(input_size, output_size)
        self.cost_function = MeanSquaredError()

        # Previous input
        self.input = None
        # Previous output
        self.output = None
        # Previous networkLoss
        self.loss = None

        # Stores whether to accumulate gradients
        self.grad = grad
        # Stores gradients if accumulated
        # Gradients are stored as a tuple (w_ih_grad, w_hh_grad, b_grad)
        if self.grad:
            self.accumulated = list()
            w_ih_grad_init = torch.zeros((self.input_size, self.hidden_size), dtype=float32)
            w_hh_grad_init = torch.zeros((self.hidden_size, self.hidden_size), dtype=float32)
            b_grad_init = torch.zeros(self.hidden_size, dtype=float32)
            self.accumulated.append((w_ih_grad_init, w_hh_grad_init, b_grad_init))
        else:
            self.accumulated = None

    def forward_prop(self, sequence: Tensor) -> Tensor:
        self.input.append(torch.squeeze(sequence))
        for entry in self.input[-1]:
            ih = torch.matmul(entry, self.w_ih)
            hh = torch.matmul(self.hidden[-1], self.w_hh)
            result = self.tanh(ih + hh + self.b)
            self.hidden.append(result)

            if self.grad:
                # Gradients are stored as a tuple (act_grad, w_ih_grad, w_hh_grad, b_grad)
                act_grad = self.tanh_prime(result)
                dp_w_ih = torch.matmul(sequence[-2], self.accumulated[-1][0].T)
                dp_w_ih = torch.matmul(dp_w_ih, self.w_hh.T)
                w_ih_grad = torch.matmul(act_grad, dp_w_ih.T)
                dp_w_hh = torch.matmul(self.w_hh, self.accumulated[-1][1].mT)
                w_hh_grad = self.hidden[-1] + torch.matmul(act_grad, dp_w_hh.T)
                dp_b = act_grad + torch.matmul(self.w_hh, self.accumulated[-1][2].mT)
                b_grad = act_grad + torch.matmul(act_grad, dp_b.T)
                self.accumulated.append((w_ih_grad, w_hh_grad, b_grad))

        self.output = self.linear.forward_prop(self.hidden[-1])
        return self.output

    def get_loss(self, expected: Tensor):
        self.loss = self.cost_function.calculate_loss(actual=self.output, expected=expected)
        return self.loss
    
    def back_prop(self):
        learn_rate = 0.01
        delta = self.cost_function.back_prop()
        self.linear.back_prop(delta=delta)
        self.loss = self.linear.delta
        w_ih_final = torch.zeros((self.input_size, self.hidden_size), dtype=float32)
        w_hh_final = torch.zeros((self.hidden_size, self.hidden_size), dtype=float32)
        b_final = torch.zeros(self.hidden_size, dtype=float32)

        w_ih_grad, w_hh_grad, b_grad = self.accumulated[-1]

        w_ih_delta = w_ih_grad * loss * learn_rate
        w_hh_delta = w_hh_grad * loss * learn_rate
        b_delta = b_grad * loss * learn_rate

        self.b += b_delta
        self.w_ih += w_ih_delta
        self.w_hh += w_hh_delta

    @classmethod
    def tanh(cls, tensor: Tensor):
        double = tensor * 2
        output = (torch.exp(double) - 1) / (torch.exp(double) + 1)
        return output

    @classmethod
    def tanh_prime(cls, tensor: Tensor):
        output = torch.ones(tensor.shape) - torch.square(tensor)
        return output


if __name__ == "__main__":
    transform = torchvision.transforms.ToTensor()
    train = load_mnist(train=True, transform=transform)
    test = load_mnist(train=False, transform=transform)

    # data loader
    # train_loader = DataLoader(train, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False)

    # Create RNN
    input_dim = 28  # input dimension
    hidden_dim = 100  # hidden layer dimension
    layer_dim = 1  # number of hidden layers
    output_dim = 10  # output dimension

    model = RNNprototype(input_dim, hidden_dim, output_dim, grad=True)

    for x, y in train:
        model.forward_prop(sequence=x)

        # Encode expected into a 1-hot vector
        expected = torch.zeros(10, dtype=float32)
        expected[y] = 1
        loss = model.get_loss(expected=y)
        model.back_prop()




