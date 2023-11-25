import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from DataHandling.DataTools import load_mnist


class TorchRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(TorchRNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # Pytorch train and test sets
    transform = torchvision.transforms.ToTensor()
    train = load_mnist(train=True, transform=transform)
    test = load_mnist(train=False, transform=transform)

    batch_size = 1
    n_iters = 60000
    num_epochs = n_iters / (len(train) / batch_size)
    num_epochs = int(num_epochs)


    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Create RNN
    input_dim = 28  # input dimension
    hidden_dim = 100  # hidden layer dimension
    layer_dim = 1  # number of hidden layers
    output_dim = 10  # output dimension

    model = TorchRNN(input_dim, hidden_dim, layer_dim, output_dim)

    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    seq_dim = 28
    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            train = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)

            # Calculate softmax and ross entropy networkLoss
            loss = error(outputs, labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            count += 1

            if count % 250 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                    # Forward propagation
                    outputs = model(images)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / float(total)

                # store networkLoss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))