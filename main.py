import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from data import load_data
from NeuralNetwork import FromDictionary
import ModelParams

def alzheimers_experiment():
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
        print(perceptron.train(train_x, train_y))
        print("Beginning testing iterations.")
        print(perceptron.test(test_x, test_y))

def mnist_baseline():
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
    train_x = np.array([np.divide(sample,255) for sample in train_x])
    test_x = np.array([np.divide(sample,255) for sample in test_x])
    
    # Get output and input sizes
    n_inputs = train_x.shape[1]
    n_outputs = train_y.shape[1]

    for entry in ModelParams.mnist_param_list:
        perceptron = FromDictionary(entry,"mnist_test_results.txt")

        print("Beginning training iterations.")
        print(perceptron.train(train_x, train_y))
        print("Beginning testing iterations.")
        print(perceptron.test(test_x, test_y))

if __name__ == '__main__':
    mnist_baseline()
