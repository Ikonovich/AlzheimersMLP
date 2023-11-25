import pandas as pd
import torch
from torch import float64, float32

from Layers.ActivationLayers.ReluLayer import ReluLayer
from Layers.DenseLinear import DenseLinear
from Layers.ManipulationLayers.DropoutLayer import DropoutLayer
from NeuralNetwork import NeuralNetwork
from Tests.TestRunner import prep_mnist


def linear_mnist():
    (train_x, train_y), (test_x, test_y), labels = prep_mnist(subset=False)

    perceptron = NeuralNetwork(
        learning_function="constant",
        loss_function="MSE",
        lrn_rate_modifier=0.01,
        output_filename="rnn_mnist.txt",
        labels_in=labels)

    input_layer = DenseLinear(output_features=32, input_features=28 * 28, bias_modifier=0.0)
    relu_one = ReluLayer(input_shape=input_layer.output_shape, previous_layer=input_layer)
    hidden_layer = DenseLinear(output_features=16, input_features=32, bias_modifier=0.00)
    relu_two = ReluLayer(input_shape=hidden_layer.output_shape, previous_layer=hidden_layer)
    dropout_layer = DropoutLayer(drop_ratio=0.5, input_shape=relu_two.output_shape, previous_layer=relu_two)
    output_layer = DenseLinear(output_features=10, input_features=16, bias_modifier=0.0)

    perceptron.add_completed_layer(input_layer)
    perceptron.add_completed_layer(relu_one)
    perceptron.add_completed_layer(hidden_layer)
    perceptron.add_completed_layer(relu_two)
    perceptron.add_completed_layer(dropout_layer)
    perceptron.add_completed_layer(output_layer)
    print("Beginning training iterations.")
    # if k_folds == True:
    #     perceptron.k_folds(train_x, train_y, labels=labels, k=k)
    # else:
    perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)

    print(f"Testing Result: {test_result[2]}")


def linear_regression_homicide():
    labels = ["rate"]

    perceptron = NeuralNetwork(
        learning_function="constant",
        loss_function="MSE",
        lrn_rate_modifier=0.01,
        output_filename="linear_homicide.txt",
        labels_in=labels)

    input_layer = DenseLinear(output_features=1, input_features=1, bias_modifier=1.0)
    perceptron.add_completed_layer(input_layer)

    data = pd.read_csv("../DataHandling/Datasets/homicide.csv")

    train_x = torch.from_numpy(data.age.values[5:]).type(float32)
    train_y = torch.from_numpy(data.num_homicide_deaths.values[5:]).type(float32)

    test_x = torch.from_numpy(data.age.values[:5])
    test_y = torch.from_numpy(data.num_homicide_deaths.values[:5])

    perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)


def recurrent_mnist():
    (train_x, train_y), (test_x, test_y), labels = prep_mnist(subset=False)

    perceptron = NeuralNetwork(
        learning_function="constant",
        loss_function="MSE",
        lrn_rate_modifier=0.01,
        output_filename="rnn_mnist.txt",
        labels_in=labels)

    input_layer = DenseLinear(output_features=32, input_features=28 * 28, bias_modifier=0.0)
    relu_one = ReluLayer(input_shape=input_layer.output_shape, previous_layer=input_layer)
    hidden_layer = DenseLinear(output_features=16, input_features=32, bias_modifier=0.00)
    relu_two = ReluLayer(input_shape=hidden_layer.output_shape, previous_layer=hidden_layer)
    dropout_layer = DropoutLayer(drop_ratio=0.5, input_shape=relu_two.output_shape, previous_layer=relu_two)
    output_layer = DenseLinear(output_features=10, input_features=16, bias_modifier=0.0)

    perceptron.add_completed_layer(input_layer)
    perceptron.add_completed_layer(relu_one)
    perceptron.add_completed_layer(hidden_layer)
    perceptron.add_completed_layer(relu_two)
    perceptron.add_completed_layer(dropout_layer)
    perceptron.add_completed_layer(output_layer)
    print("Beginning training iterations.")
    # if k_folds == True:
    #     perceptron.k_folds(train_x, train_y, labels=labels, k=k)
    # else:
    perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)

    print(f"Testing Result: {test_result[2]}")

if __name__ == "__main__":
    # linear_mnist()
    linear_regression_homicide()
