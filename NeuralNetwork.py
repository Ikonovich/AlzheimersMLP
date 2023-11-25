import copy
import datetime
import time
from collections import deque
from typing import Callable

from Layers.Layer import Layer
from Layers.DenseLinear import DenseLinear
from Layers.ConvolutionLayer import ConvolutionalLayer
from Layers.LossLayers.MeanSquaredError import MeanSquaredError

from Math import Functions

from keras.datasets import mnist

import numpy as np

import Metrics
import ModelParams
from Process import shuffle, split, kfold_merge
from data import load_data

# Represents a neural network composed of layers
# learning_rate: String. constant, linear_decreasing
# lrn_rate_modifier: Int. Determines strength/rate of growth of learning rate function
class NeuralNetwork:

    def __init__(
            self,
            learning_function="constant",
            lrn_rate_modifier=0.05,
            loss_function="MSE",
            output_filename="test_results.txt",
            labels_in=None,
            batch_size=500,
            test_size=100):

        # Stores the networkLoss function and previous networkLoss
        self.loss = None
        self.loss_layer = None
        self.set_loss_function(loss_function)
        # Stores the best calculated validation networkLoss during training with k-folds or batches
        self.validation_loss = float('inf')
        # Stores the accuracy from the state with the lowest validation networkLoss.
        self.accuracy = 0

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
        time_start = time.time_ns()
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
            results = self.test(x_merged, y_merged, labels=labels)

        print(f"K-Folds total training time: {(time.time_ns() - time_start) / 1e9}")
        print(f"Displaying results from best saved state, which comes from k-folds iteration {saved_state_batch}.")
        return self.saved_state[1].retrieve_results(save=True)

    # Takes a list of inputs and trains the network with them
    def train(self, x_set, y_set, labels, detailed_output=False):
        print("Running train sequence.")
        time_start = time.time_ns()
        results = self.run(x_set, y_set, labels, training=True)
        print(f"Train sequence time: {(time.time_ns() - time_start) / 1e9}")
        return results

    # Takes a list of inputs and tests the network with them,
    # without performing backprop
    def test(self, x_set, y_set, labels, training=False):
        print("Running test sequence.")
        time_start = time.time_ns()
        results = self.run(x_set, y_set, labels, training=False)
        print(f"Test sequence time: {(time.time_ns() - time_start) / 1e9}")
        return results

    # Runs through a full data set, either with training or not.
    def run(self, x_list, y_list, labels, training):
        self.is_training = training

        loss = None

        # Update the training status on every layer
        for layer in self.layers:
            layer.training = self.is_training

        self.labels = labels

        # Clear results of previous run
        self.clear_results()
        # Save the size of the input data set
        self.sample_count = len(x_list)

        start = time.time()
        i = 0
        batch_correct = 0
        last_ten_batch_accuracies = deque(maxlen=10)

        outputs = list()
        expected = list()
        for (sample, value) in zip(x_list, y_list):

            output = self.forward_prop(sample, value)
            outputs.append(output)
            expected.append(value)
            if np.argmax(output) == np.argmax(value):
                self.correct += 1
                batch_correct += 1
            else:
                self.wrong += 1
            if training == True:
                self.back_prop()

            # Every batch, update the percent complete and batch accuracy.
            # We also clear the output record and save the state if training
            i += 1
            if i % self.batch_size == 0:
                self.percent_complete = (i / self.sample_count) * 100
                self.last_batch_accuracy = batch_correct/1000
                last_ten_batch_accuracies.append(self.last_batch_accuracy)

                print(f"{self.percent_complete}%: Batch {i}: Batch Accuracy: {self.last_batch_accuracy * 100}%",
                      end="\r")
                batch_correct = 0

                if training == True:
                    loss = self.loss_layer(outputs, expected)

                    if self.validation_loss > loss:
                        print(f"Saving state from batch {i} with networkLoss {loss} and accuracy {self.last_batch_accuracy}.\n"
                              f"Previously saved state had networkLoss {self.validation_loss} and accuracy {self.accuracy}.")
                        self.save_state()
                        self.validation_loss = loss
                        self.accuracy = self.last_batch_accuracy
                        saved_state_batch = i



                    outputs = list()
                    expected = list()

        print(f"Time elapsed: {time.time() - start}")
        accuracy = 1 - (self.wrong / (self.wrong + self.correct))
        print(f"Final Accuracy: {accuracy}")

        return i, accuracy, self.retrieve_results(save=False), loss

    # Tests for convergence. Returns true if convergence is anticipated.
    def test_convergence(self):
        pass

    # Computes one iteration of forward prop for the network
    def forward_prop(self, x_sample, y_value=None, grad=True):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                # Run the first layer using the input data
                layer.forward_prop(x_sample)
            else:
                # Run every layer after the input layer using the output of the previous layer
                layer.forward_prop(self.layers[i - 1].output)

        result = self.layers[len(self.layers) - 1].output
        # If y value isn't none, gets the error value of the result and stores it.
        if y_value is not None:

            self.results.append(result)
            self.expected.append(y_value)

            # Calculate the networkLoss and networkLoss gradient
            self.loss = self.loss_layer.calculate_loss(actual=result, expected=y_value, grad=grad)
        return result

    # Computes one iteration of backward prop for the network
    # First gets the current learning rate so it can pass it to each
    # layer
    def back_prop(self):

        lrn_rate = self.get_learn_rate()
        delta = self.loss * self.loss_layer.prime
        for i in range(len(self.layers)):
            # Start from the bottom layer, which is at the last index
            layer = self.layers[len(self.layers) - (i + 1)]

            # Learn rate is applied here.
            delta = layer.back_prop(delta=delta * lrn_rate)

    # Takes a single input and returns the response of the network.
    def predict(self, x_sample):
        return self.forward_prop(x_sample)

    # Adds a layer to the network with the provided parameters.
    # activation: String. Relu, sigmoid, tanh, leaky_relu
    # output_size: Int.
    # input_size: Int. Only required for the first layer of the network.
    # All layers after the first layer have their input sizes computed from the output
    # of the previous layer.
    def add_layer(self, activation_string, bias_lrn_modifier=0, dropout_modifier=0, input_shape=None, output_shape=None):

        # Unless this is the first layer, get the length of the last layer and set
        # its output size to be the input size to the next layer.
        if len(self.layers) > 0:
            previous_layer = self.layers[len(self.layers) - 1]

            new_layer = DenseLinear(output_features=output_shape, input_features=previous_layer.output_shape,
                                    bias_modifier=bias_lrn_modifier, previous_layer=previous_layer)

            # Set the reference in the previous layer
            previous_layer.next_layer = new_layer
            self.layers.append(new_layer)

        else:
            if input_shape is None:
                raise Exception("The input size of the first layer must be provided.")
            self.layers.append(
                DenseLinear(output_features=output_shape, input_features=input_shape,
                            bias_modifier=bias_lrn_modifier))

    # Adds an already constructed layer to the network in the last position.
    def add_completed_layer(self, new_layer: Layer):

        # Unless this is the first layer, get the length of the last layer and set
        # its output size to be the input size to the next layer, as well
        # as connecting the this layer and the layer before it.
        if len(self.layers) > 0:
            previous_layer = self.layers[len(self.layers) - 1]

            new_layer.set_input_shape(previous_layer.output_shape)
            new_layer.next_layer = previous_layer
            previous_layer.next_layer = new_layer

        self.layers.append(new_layer)

    # Sets the networkLoss function
    def set_loss_function(self, func_name: str):
        if func_name is "MSE":
            self.loss_layer = MeanSquaredError()
        else:
            raise ValueError("The provided networkLoss function is not valid.")

    # Used to retrieve the learn rate for each iteration of backprop
    def get_learn_rate(self):
        return self.learning_function(self)

    # Returns the input size
    def get_input_size(self):
        input_layer = self.layers[0]
        return input_layer.input_shape

    # Returns the output size
    def get_output_size(self):
        output_layer = self.layers[len(self.layers) - 1]
        return output_layer.output_shape

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
        for layer in self.layers:
            layer.save()


# Creates a neural network from a dictionary as defined in ModelParams.py
def FromDictionary(params_in, output_filename, input_size):

    netparams = params_in["network"]
    lrn_rate = netparams["learning_rate"]
    modifier = netparams["lrn_rate_modifier"]

    network = NeuralNetwork(lrn_rate, modifier, output_filename)
    layers = params_in["layers"]

    for layer in layers:
        activation_string = layer.get("activation")
        num_outputs = layer.get("n_outputs")
        bias_modifier = layer.get("bias")

        # Input size is always passed but the network only uses it if it's the first layer in the network.
        network.add_layer(
            activation_string=activation_string, input_shape=input_size,
            output_shape=num_outputs, bias_lrn_modifier=bias_modifier,
            dropout_modifier=0)

    return network


if __name__ == "__main__":

    # layerOne = ConvolutionalLayer((28, 28), 4, (5, 5), (2, 2))

    # network = NeuralNetwork()

    pass
