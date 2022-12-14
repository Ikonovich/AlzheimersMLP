import copy
import datetime
import time
from collections import deque

from numba import int32, float32
from numba.experimental import jitclass

from Math import Functions

from keras.datasets import mnist

import numpy as np

import Metrics
import ModelParams
from Layers.DenseLayer import DenseLayer
from Process import shuffle, split, kfold_merge
from data import load_data

# Represents a neural network composed of layers
# learning_rate: String. constant, linear_decreasing
# lrn_rate_modifier: Int. Determines strength/rate of growth of learning rate function
class NeuralNetwork:

    def __init__(
            self,
            learning_function,
            lrn_rate_modifier,
            output_filename="test_results.txt",
            labels_in=None,
            batch_size=500,
            test_size=100):

        # The size of the batch training set.
        self.batch_size = batch_size
        # The size of the batch testing set.
        self.test_size = test_size
        # A list of tuples containing the results of each testing batch.
        # Each tuple contains (batch_number, f1_score)
        self.batch_results = []

        # Stores the best recorded test accuracy since it was last zeroed out
        # Used by k-means and run
        self.best_test_accuracy = 0

        # Stores the last recorded training batch accuracy
        self.last_batch_accuracy = 0

        # Stores whether we are currently training
        self.is_training = False

        self.labels = labels_in
        self.learning_function = Functions.LEARNING_FUNCTION_MAP[learning_function]
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

        # Saves the accuracy and state of the network after the highest validation testing value
        # during a run of k-folds.
        self.saved_state = (0, self)

    # Runs k-folds cross validation
    def k_folds(self, x_set, y_set, labels=None, k=10):
        x_set, y_set = shuffle(x_set, y_set)

        # Split the data
        x_folds, y_folds = split(x_set, y_set, k)


        # Store last test accuracy and results
        last_test_accuracy = 0
        final_results = None
        # Stores the iteration of k-folds when the state was saved
        saved_state_batch = 0

        #### to do
        # Create the folds and run them
        for i in range(len(x_folds)):
            x_merged, y_merged = kfold_merge(x_folds, y_folds, i)

            print(f"Running training with dropout fold {i}")
            self.train(x_merged, y_merged, labels=labels)


            ###### TODO: FIX
            print(f"Running testing on fold {i}")
            accuracy = self.test(x_merged, y_merged, labels=labels)
            if self.last_test_accuracy > self.best_test_accuracy:
                self.save_state()
                saved_state_batch = i

        print(f"Displaying results from best saved state, which comes from k-folds iteration {saved_state_batch}.")
        return self.saved_state[2].retrieve_results(save=True)

    # Takes a list of inputs and trains the network with them
    def train(self, x_set, y_set, labels, detailed_output=False):
        return self.run(x_set, y_set, labels, training=True)

    # Takes a list of inputs and tests the network with them,
    # without performing backprop
    def test(self, x_set, y_set, labels, training=False):
        results = self.run(x_set, y_set, labels, training=False)
        self.last_test_accuracy = results[1]
        return results

    # Runs through a full data set, either with training or not.
    def run(self, x_list, y_list, labels, training):
        self.is_training = training
        self.labels = labels

        # Clear results of previous run
        self.clear_results()
        # Save the size of the input data set
        self.sample_count = len(x_list)

        start = time.time()
        i = 0
        batch_correct = 0
        last_ten_batch_accuracies = deque(maxlen=10)

        for (sample, value) in zip(x_list, y_list):

            output = self.forward_prop(sample, value)
            if output == True:
                batch_correct += 1
            if training == True:
                self.back_prop(value)

            # Every batch, update the percent complete and batch accuracy.
            i += 1
            if i % self.batch_size == 0:
                self.percent_complete = (i / self.sample_count) * 100
                self.last_batch_accuracy = batch_correct/1000
                last_ten_batch_accuracies.append(self.last_batch_accuracy)

                print(f"{self.percent_complete}%: Batch {i}: Batch Accuracy: {self.last_batch_accuracy * 100}%",
                      end="\r")
                batch_correct = 0

        print(f"Time elapsed: {time.time() - start}")
        accuracy = 1 - (self.wrong / (self.wrong + self.correct))
        print(f"Final Accuracy: {accuracy}")

        return i, accuracy, self.retrieve_results(save=False)

    # Tests for convergence. Returns true if convergence is anticipated.
    def test_convergence(self):
        pass

    # Computes one iteration of forward prop for the network
    def forward_prop(self, x_sample, y_value=None):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                # Run the first layer using the input data
                layer.forward_prop(x_sample)
            else:
                # Run every layer after the input layer using the output of the previous layer
                layer.forward_prop(self.layers[i - 1].last_output)

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


    # Computes one iteration of backward prop for the network
    # First gets the current learning rate so it can pass it to each
    # layer
    def back_prop(self, expected):

        lrn_rate = self.get_learn_rate()

        for i in range(len(self.layers)):
            # Start from the bottom layer, which is at the last index
            layer = self.layers[len(self.layers) - (i + 1)]

            # If i == 0, do an output backprop:
            if i == 0:
                layer.output_back_prop(expected, learn_rate=lrn_rate)
            else:
                layer.hidden_back_prop(learn_rate=lrn_rate)


    # Takes a single input and returns the response of the network.
    def predict(self, x_sample):
        return self.forward_prop(x_sample)

    # Adds a layer to the network with the provided parameters.
    # activation: String. Relu, sigmoid, tanh, leaky_relu
    # output_size: Int.
    # input_size: Int. Only required for the first layer of the network.
    # All layers after the first layer have their input sizes computed from the output
    # of the previous layer.
    def add_layer(self, activation, output_size, bias_lrn_modifier=0, dropout_modifier=0, input_size=None):

        # Use the activation string to get the activation function and derivative
        # from the map
        activation_func = Functions.ACTIVATION_FUNCTION_MAP[activation][0]
        derivative_func = Functions.ACTIVATION_FUNCTION_MAP[activation][1]

        # Unless this is the first layer, get the length of the last layer and set
        # its output size to be the input size to the next layer.
        if len(self.layers) > 0:
            previous_layer = self.layers[len(self.layers) - 1]

            new_layer = DenseLayer(
                activation=activation_func,
                derivative=derivative_func,
                input_size=previous_layer.output_size,
                output_size=output_size,
                bias_lrn_modifier=bias_lrn_modifier,
                dropout_modifier=dropout_modifier)

            # Set the reference to the previous layer
            new_layer.previous_layer = previous_layer
            previous_layer.next_layer = new_layer
            self.layers.append(new_layer)

        # If this is the first layer, input_size must be provided.
        else:
            if input_size is None:
                raise Exception("The input size of the first layer must be provided.")
            self.layers.append(
                DenseLayer(
                    activation=activation_func, derivative=derivative_func,
                    input_size=input_size, output_size=output_size,
                    bias_lrn_modifier=bias_lrn_modifier,
                    dropout_modifier=dropout_modifier))


    # Used to retrieve the learn rate for each iteration of backprop
    def get_learn_rate(self):
        return self.learning_function(self)

    # Returns the input size
    def get_input_size(self):
        input_layer = self.layers[0]
        return input_layer.input_size

    # Returns the output size
    def get_output_size(self):
        output_layer = self.layers[len(self.layers) - 1]
        return output_layer.output_size

    # Clears results from the previous run
    def clear_results(self):
        self.correct = 0
        self.wrong = 0
        self.results = []
        self.expected = []

    # Returns architecture details and results of last run
    def retrieve_results(self, save=False):

        print("Getting results...")

        returnStr = f"\nRun finished at {datetime.datetime.now()}\n"
        returnStr +=  f"--- Parameters --- " \
                + f"\nLearning rate: {self.learning_function.__name__}" \
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

        if save == True:
            with open(self.output_filename, 'a') as f:
                f.write(returnStr)

        return returnStr

    # Used to record the state of the network at a specific time
    def save_state(self):
        self.saved_state = (self.last_test_accuracy, copy.deepcopy(self))


# Creates a neural network from a dictionary as defined in ModelParams.py
def FromDictionary(params_in, output_filename, input_size):

    netparams = params_in["network"]
    lrn_rate = netparams["learning_rate"]
    modifier = netparams["lrn_rate_modifier"]

    network = NeuralNetwork(lrn_rate, modifier, output_filename)
    layers = params_in["layers"]

    for layer in layers:
        activation = layer.get("activation")
        num_outputs = layer.get("n_outputs")
        bias_modifier = layer.get("bias")

        # Input size is always passed but the network only uses it if it's the first layer in the network.
        network.add_layer(
            activation=activation, input_size=input_size,
            output_size=num_outputs, bias_lrn_modifier=bias_modifier,
            dropout_modifier=0)

    return network


# Run the alzheimers dataset through a bunch of models
def run_alzheimers(print_result=False, k_folds=False, k=10):

    train_x, train_y, test_x, test_y = load_data(1.0)
    n_inputs = train_x.shape[1]
    n_outputs = train_y.shape[1]
    labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

    train_x = np.array([np.divide(sample,255) for sample in train_x])
    test_x = np.array([np.divide(sample,255) for sample in test_x])

    for entry in ModelParams.alzheimers_param_list:
        perceptron = FromDictionary(entry,"alzheimers_test_results.txt", input_size=n_inputs)

        print("Beginning training iterations.")
        if k_folds == True:
            perceptron.k_folds(train_x, train_y, labels=labels, k=k)
        else:
            perceptron.train(train_x, train_y, labels=labels)
        print("Beginning validation iterations.")
        test_result = perceptron.test(test_x, test_y, labels=labels)

        if print_result == True:
            print(f"Testing Result: {test_result}")


def run_mnist(print_result=False, k_folds=False, k=10):

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
        perceptron = FromDictionary(entry, "mnist_test_results.txt", input_size=n_inputs)

        print("Beginning training iterations.")
        if k_folds == True:
            perceptron.k_folds(train_x, train_y, labels=labels, k=k)
        else:
            perceptron.train(train_x, train_y, labels=labels)
        print("Beginning validation iterations.")
        test_result = perceptron.test(test_x, test_y, labels=labels)

        if print_result == True:
            print(f"Testing Result: {test_result[2]}")

if __name__ == "__main__":

    #run_alzheimers(print_result=True, k_folds=True, k=10)

    run_mnist(print_result=True, k_folds=False)

