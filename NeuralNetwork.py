import datetime
import math as math
import sys
import time
import Functions

import tensorflow
from keras.datasets import mnist

import numpy as np

import Metrics
import ModelParams
from data import load_data
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support


# Represents a layer of a neural network
class Layer:
    def __init__(self, network, activation, input_size, output_size):


        # defines the drop-out rate
        self.dropout = None

        # Stores the network that this layer is a part of
        self.network = network

        # Stores the previous layer in the network, if any
        self.previous_layer = None
        # Stores the following layer in the network, if any
        self.next_layer = None

        # Used to show results
        self.activation_string = activation

        self.activation = Functions.ACTIVATION_FUNCTION_MAP[activation][0]
        self.derivative = Functions.ACTIVATION_FUNCTION_MAP[activation][1]
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)

        # Stores the last provided input
        self.last_input = None
        # Stores the last calculated output
        self.last_output = None
        # Stores the last calculated delta
        self.last_delta = None

        # Stores the per-neuron bias
        self.bias = np.zeros(output_size)

    # Handles forward propagation when given input data
    # The size of data_in must match the input size of this layer
    def forward_prop(self, data_in=None):

        # If data_in isn't None, this should be an input layer.
        input_in = data_in
        if data_in is None:
            input_in = self.previous_layer.last_output

        if len(input_in) != self.input_size:
            raise Exception("Invalid input size to layer.forward_prop.")

        self.last_input = input_in
        self.last_output = (np.dot(input_in, self.weights) / self.input_size) + self.bias
        return self.last_output

    def get_output(self):
        return self.last_output

    # Performs backprop for the output layer. Takes in a list of expected results.
    def output_back_prop(self, expected):
        higher_layer = self.previous_layer

        result = [0] * self.output_size
        result[np.argmax(self.last_output)] = 1
        result = np.asarray(result)
        y = np.asarray(expected)

        # The results are the last output of this layer
        error = -(expected - self.last_output)
        prime = self.derivative(self.last_output) #.flatten()

        self.last_delta = error * prime
        self.back_prop(prime)

    def hidden_back_prop(self):
        lower_layer = self.next_layer
        prime = self.derivative(self.last_output).flatten()
        self.last_delta = np.dot(lower_layer.last_delta, lower_layer.weights.T) * prime
        self.back_prop(prime)

    def back_prop(self, prime):
        adjustedInput = self.last_input[np.newaxis]
        adjustedDelta = self.last_delta[np.newaxis]
        weight_updates = np.dot(adjustedInput.T, adjustedDelta * self.network.learning_function(self.network))
        bias_updates = self.last_delta * self.network.learning_function(self.network) * 0.1
        self.weights -= weight_updates
        self.bias -= bias_updates

    def get_layer_string(self):
        return f"\nLayer Input Size: {self.input_size}" \
            + f"\nLayer Output Size: {self.output_size}" \
            + f"\nLayer Activation: {self.activation_string}\n" \
            + f"\nDropout Rate: None\n"

# Represents a neural network composed of layers
# learning_rate: String. constant, linear_decreasing
# lrn_rate_modifier: Int. Determines strength/rate of growth of learning rate function
class NeuralNetwork:

    def __init__(self, learning_rate, lrn_rate_modifier, output_filename="test_results.txt", labels_in=None):

        # Stores the last recorded batch accuracy
        self.last_batch_accuracy = 0

        # Stores whether we are currently training
        self.is_training = False

        self.labels = labels_in
        self.learning_function = Functions.LEARNING_FUNCTION_MAP[learning_rate]
        self.lrn_rate_modifier = lrn_rate_modifier
        self.layers = []

        # Stores the size of the dataset and the percent complete during a run() call.
        # Used primarily for updating dynamic learning rates
        self.sample_count = None
        self.percent_complete = None


        # Stores the last set of results produced by the run() function,
        # as called by train or test.
        self.correct = 0
        self.wrong = 0
        self.results = []
        self.expected = []

        # Stores the output file of the run's results
        self.output_filename = output_filename


    # Takes a list of inputs and trains the network with them
    def train(self, x_set, y_set, labels, detailed_output=False):
        return self.run(x_set, y_set, labels, training=True)

    # Takes a list of inputs and tests the network with them,
    # without performing backprop
    def test(self, x_set, y_set, labels, training=False):
        return self.run(x_set, y_set, labels, training=False)

    # Runs through a full data set, either with training or not.
    def run(self, x_set, y_set, labels, training):
        self.is_training = training
        self.labels = labels

        # Clear results of previous run
        self.clear_results()
        # Save the size of the input data set
        self.sample_count = len(x_set)

        start = time.time()
        i = 0
        batch_correct = 0
        for (sample, value) in zip(x_set, y_set):
            output = self.forward_prop(sample, value)
            if output == True:
                batch_correct += 1
            if training == True:
                self.back_prop(value)

            # Every batch, update the percent complete and batch accuracy.
            i += 1
            if i % 1000 == 0:
                self.percent_complete = (i / self.sample_count) * 100
                self.last_batch_accuracy = batch_correct/1000
                print(f"{self.percent_complete}%: Batch {i}: Batch Accuracy: {self.last_batch_accuracy * 100}%",
                      end="\r")
                batch_correct = 0

        print(f"Time elapsed: {time.time() - start}")
        accuracy = 1 - (self.wrong / (self.wrong + self.correct))
        print(f"Final Accuracy: {accuracy}")

        return self.save_results()

    # Computes one iteration of forward prop for the network
    def forward_prop(self, x_sample, y_value=None):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                # Run the first layer using the input data
                layer.forward_prop(x_sample)
            else:
                # Run every layer after the input layer using the output of the previous layer
                layer.forward_prop()

        result = self.layers[len(self.layers) - 1].last_output
        # If y value isn't none, gets the error value of the result and stores it.
        if y_value is not None:

            self.results.append(result)
            self.expected.append(y_value)

            if np.argmax(result) == np.argmax(y_value):
                self.correct += 1
                return True
            else:
                self.wrong += 1
                return False


        # Return the output of the last layer as our final result of the computation


    # Computes one iteration of backward prop for the network
    def back_prop(self, expected):
        for i in range(len(self.layers)):
            # Start from the bottom layer, which is at the last index
            layer = self.layers[len(self.layers) - (i + 1)]

            # If i == 0, do an output backprop:
            if i == 0:
                layer.output_back_prop(expected)
            else:
                layer.hidden_back_prop()


    # Takes a single input and returns the response of the network.
    def predict(self, x_sample):
        return self.forward_prop(x_sample)

    # Adds a layer to the network with the provided parameters.
    # activation: String. Relu, sigmoid, tanh, leaky_relu
    # output_size: Int.
    # input_size: Int. Only required for the first layer of the network.
    # All layers after the first layer have their input sizes computed from the output
    # of the previous layer.
    def add_layer(self, activation, output_size, input_size=None):
        # Unless this is the first layer, get the length of the last layer and set
        # its output size to be the input size to the next layer.
        if len(self.layers) > 0:
            previous_layer = self.layers[len(self.layers) - 1]
            new_layer = Layer(self, activation, previous_layer.output_size, output_size)

            # Sets up the nodal links between the layers
            new_layer.previous_layer = previous_layer
            previous_layer.next_layer = new_layer

            self.layers.append(new_layer)

        # If this is the first layer, input_size must be provided.
        else:
            if input_size is None:
                raise Exception("The input size of the first layer must be provided.")
            self.layers.append(Layer(self, activation, input_size, output_size))

    # Clears results from the previous run
    def clear_results(self):
        self.correct = 0
        self.wrong = 0
        self.results = []
        self.expected = []


    # Returns the input size
    def get_input_size(self):
        input_layer = self.layers[0]
        return input_layer.input_size

    # Returns the output size
    def get_output_size(self):
        output_layer = self.layers[len(self.layers) - 1]
        return output_layer.output_size

    # Returns architecture details and results of last run
    def save_results(self):

        print("Getting results...")

        returnStr = f"\nRun finished at {datetime.datetime.now()}\n"
        returnStr +=  f"--- Parameters --- " \
                + f"\nLearning rate: {self.learning_function}" \
                + f"\nNumber of Hidden Layers: {len(self.layers)} \n\nLayers:\n"

        for i in range(len(self.layers)):
            returnStr += f"Layer {i}: "
            if i == 0:
                returnStr += "Input Layer"
            elif i == len(self.layers) - 1:
                returnStr += "Output Layer"
            else:
                returnStr += "Hidden Layer"
            returnStr += f"\nParameters: \n{self.layers[i].get_layer_string()}"


        metrics = Metrics.get_metrics(self.results, self.expected, self.labels)
        returnStr += f"\n\n--- Results --- "

        for i in range(len(metrics)):
            entry = metrics[i]
            if i == 0:
                returnStr += "\n---Network Results---\n"
            elif i == 1:
                returnStr += "\n----Class Results---- \n"
            else:
                returnStr += "\n--------------------- \n"
            for key in entry:
                value = entry[key]
                returnStr += f"{key}: {value}\n"

        with open(self.output_filename, 'w') as f:
            f.write(returnStr)

        return returnStr



# Creates a neural network from a dictionary as defined in ModelParams.py
def FromDictionary(params_in, output_filename):

    netparams = params_in["network"]
    lrn_rate = netparams["learning_rate"]
    modifier = netparams["lrn_rate_modifier"]

    network = NeuralNetwork(lrn_rate, modifier, output_filename)
    layers = params_in["layers"]

    for layer in layers:
        activation = layer.get("activation")
        num_inputs = layer.get("n_inputs")
        num_outputs = layer.get("n_outputs")

        network.add_layer(activation, num_outputs, input_size=num_inputs)

    return network

    #     "network": {
    #         "learning_rate": "constant",
    #         "lrn_rate_modifier": 0.01,
    #         "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    #     },
    #     "layers": [
    #         {"activation": "relu",
    #          "size": 256,
    #          "n_inputs": 36100},
    #         {"activation": "relu",
    #          "size": 64},
    #         {"activation": "sigmoid",
    #          "size": 32,
    #          "n_outputs": 4}
    #     ]
    #
    # }

# Run the alzheimers dataset through a bunch of models
def run_alzheimers(print_result=False):

    train_x, train_y, test_x, test_y = load_data(1.0)
    n_inputs = train_x.shape[1]
    n_outputs = train_y.shape[1]
    labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

    train_x = np.array([np.divide(sample,255) for sample in train_x])
    test_x = np.array([np.divide(sample,255) for sample in test_x])

    print(train_x[0])

    for entry in ModelParams.alzheimbers_param_list:
        perceptron = FromDictionary(entry,"alzheimers_test_results.txt")

        print("Beginning training iterations.")
        train_result = perceptron.train(train_x, train_y, labels=labels)
        print("Beginning testing iterations.")
        test_result = perceptron.test(test_x, test_y, labels=labels)

        if print_result == True:
            print(f"Training Result: {train_result}")
            print(f"Testing Result: {test_result}")


def run_mnist(print_result=False):

    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Convert y from integers into one-hot encoding
    train_y_arr = []
    for val in train_y:
        newVec = [0] * 10
        newVec[int(val)] = 1
        train_y_arr.append(newVec)
    train_y = np.asarray(train_y_arr)

    test_y_arr = []
    for val in test_y:
        newVec = [0] * 10
        newVec[val] = 1
        test_y_arr.append(newVec)
    test_y = np.asarray(test_y_arr)

    # Flatten image arrays
    train_x = [sample.flatten() for sample in train_x]
    test_x = [sample.flatten() for sample in test_x]

    # normalize values between 0 and 1
    train_x = np.array([np.divide(sample, 255) for sample in train_x])
    test_x = np.array([np.divide(sample, 255) for sample in test_x])

    # Get output and input sizes
    n_inputs = train_x.shape[1]
    n_outputs = train_y.shape[1]

    labels = [i for i in range(0, 10)]

    for entry in ModelParams.mnist_param_list:
        perceptron = FromDictionary(entry, "mnist_test_results.txt")

        print("Beginning training iterations.")
        train_result = perceptron.train(train_x, train_y, labels=labels)
        print("Beginning testing iterations.")
        test_result = perceptron.test(test_x, test_y, labels=labels)

        if print_result == True:
            print(f"Training Result: {train_result}")
            print(f"Testing Result: {test_result}")

if __name__ == "__main__":

    # run_alzheimers()
    run_mnist()


