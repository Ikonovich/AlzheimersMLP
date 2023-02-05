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

    for entry in ModelParams.alzheimers_param_list:
        perceptron = FromDictionary(entry,"alzheimers_test_results.txt")

        print("Beginning training iterations.")
        print(perceptron.train(train_x, train_y))
        print("Beginning testing iterations.")
        print(perceptron.test(test_x, test_y))



if __name__ == '__main__':
    pass
