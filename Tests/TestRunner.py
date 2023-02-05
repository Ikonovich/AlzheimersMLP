import numpy as np
from keras.datasets import mnist

import ConvolutionTests as conv
import ModelParams
import PoolingTest as pool
from Experimental.ConvolutionPrototype import ConvolutionPrototype

from Layers.ConvolutionLayer import ConvolutionalLayer
from Layers.DenseLayer import DenseLayer
from NeuralNetwork import FromDictionary, NeuralNetwork

# This module stores a variety of tests for use with the NeuralNetwork class and associated files.


# Run the alzheimers dataset through a bunch of models
from data import load_data


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

    #### REMOVED TO TEST 2D INPUT HANDLING
    # Flatten image arrays
    # train_x = [sample.flatten() for sample in train_x]
    # test_x = [sample.flatten() for sample in test_x]

    # normalize values between 0 and 1
    train_x = np.array([np.divide(sample, 255) for sample in train_x])
    test_x = np.array([np.divide(sample, 255) for sample in test_x])

    # Get output and input sizes
    n_inputs = train_x.shape[1] * train_x.shape[2]
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

def run_conv_mnist(print_result=True, k_folds=False, k=1):
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

    #### REMOVED TO TEST 2D INPUT HANDLING
    # Flatten image arrays
    # train_x = [sample.flatten() for sample in train_x]
    # test_x = [sample.flatten() for sample in test_x]

    # normalize values between 0 and 1
    train_x = np.array([np.divide(sample, 255) for sample in train_x])
    test_x = np.array([np.divide(sample, 255) for sample in test_x])

    # Get output and input sizes
    n_inputs = train_x.shape[1] * train_x.shape[2]
    n_outputs = train_y.shape[1]

    labels = [i for i in range(0, 10)]

    perceptron = NeuralNetwork(
        learning_function="constant",
        lrn_rate_modifier=0.1,
        output_filename="convmnist.txt",
        labels_in=labels)

    conv_layer = ConvolutionalLayer(
        input_shape=train_x[0].shape,
        num_filters=4,
        filter_shape=(5, 5),
        activation_string="relu")

    hidden_layer = DenseLayer(
        activation_string="relu",
        output_shape=16,
        bias_lrn_modifier=0.0,
        dropout_modifier=0)

    output_layer = DenseLayer(
        activation_string="sigmoid",
        output_shape=10,
        bias_lrn_modifier=0.0,
        dropout_modifier=0)

    perceptron.add_completed_layer(conv_layer)
    perceptron.add_completed_layer(hidden_layer)
    perceptron.add_completed_layer(output_layer)


    print("Beginning training iterations.")
    if k_folds == True:
        perceptron.k_folds(train_x, train_y, labels=labels, k=k)
    else:
        perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)

    if print_result == True:
        print(f"Testing Result: {test_result[2]}")

def run_conv_proto_mnist(print_result=True, k_folds=False, k=1):
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

    # normalize values between 0 and 1
    train_x = np.array([np.divide(sample, 255) for sample in train_x])
    test_x = np.array([np.divide(sample, 255) for sample in test_x])

    # Get output and input sizes
    n_inputs = train_x.shape[1] * train_x.shape[2]
    n_outputs = train_y.shape[1]

    # Create the test object
    cvp = ConvolutionPrototype()

    cvp = ConvolutionPrototype()
    total = 0
    correct = 0
    for sample, value in zip(train_x, train_y):
        result = cvp.iterate(data=sample)

        if np.argmax(result) == np.argmax(value):
            correct += 1

        total += 1
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100}%")
        cvp.backprop(value)

if __name__ == "__main__":

    run_conv_proto_mnist()

    # data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    # kernel = np.asarray([[1, 0], [0, 1]])
    #
    # matr = np.zeros((2, 2, 2))
    # matr[0][0][1] = 1
    # matr[0][1][0] = 1
    # print(matr)
    #
    # matr[1][0][0] = 1
    # matr[1][1][1] = 1
    #
    # print(matr)
    #
    # nums = np.asarray([[1, 2], [3, 4]])
    #
    # result = np.matmul(nums, matr)
    # print(result)
    # flat = result.flatten()
    # print(flat)
    #
    # print(flat.reshape(result.shape))
