import sys
import time
import Functions

import numpy as np
from data import load_data



class OriginalLearner:
    # hidden_activation: String. Relu, sigmoid, tanh, leaky_relu
    # output_activation: String. Relu, sigmoid, tanh, leaky_relu
    # input_size: Int.
    # output_size: Int.
    # hidden_layer_size: Int. Size of each hidden layer
    # learning_rate: String. static_learning_rate, neg_linear_learn_rate
    # lrn_rate_modifier: Int. Determines strength/rate of growth of learning rate function

    def __init__(
            self,
            hidden_activation,
            output_activation,
            input_size,
            output_size,
            hidden_layer_size,
            learning_rate,
            lrn_rate_modifier):

        # Maps activation strings to their functions and derivatives
        self.learnRateToFunctions = {
            "leaky_relu": (Functions.leaky_relu, Functions.leaky_relu_prime),
            "relu": {Functions.relu, Functions.relu_prime},
            "sigmoid": {Functions.sigmoid, Functions.sigmoid_two},
            "tanh": {Functions.tanh, Functions.tanh_prime},
            "softmax": {Functions.softmax, Functions.softmax_prime}
        }

        # Stores whether or not the network is currently training

        self.is_training = False
        print("Perceptron input size: " + str(input_size))

        self.learningRate = learning_rate
        self.lrnRateModifier = lrn_rate_modifier

        self.correct = 0
        self.wrong = 0

        self.results = []
        self.expected = []

        self.data = None

        # Set layer sizes
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size

        # Set activation functions
        self.hidden_activation = self.learnRateToFunctions[hidden_activation][0]
        self.hidden_derivative = self.learnRateToFunctions[hidden_activation][1]

        self.output_activation = self.learnRateToFunctions[output_activation][0]
        self.output_derivative = self.learnRateToFunctions[output_activation][1]

        # Set layer sizes
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size


        # create layers
        self.weightsOne = np.random.randn(input_size, hidden_layer_size)
        self.weightsTwo = np.random.randn(hidden_layer_size, hidden_layer_size)
        self.weightsThree = np.random.randn(hidden_layer_size, output_size)

        self.bOne = [0] * hidden_layer_size
        self.bTwo = [0] * hidden_layer_size
        self.bThree = [0] * output_size

    def save_results(self):

        print("Getting results...")

        predictions = []
        # Convert predictions into binary class predictions
        for vector in self.results:
            maxIndex = np.argmax(vector)
            newVec = [0] * self.output_size
            newVec[maxIndex] = 1
            predictions.append(newVec)

        expected = self.expected

        accuracy = 1 - (self.wrong / (self.wrong + self.correct))
        print("Accuracy: " + str(accuracy))
        # precision = precision_score(
        #     y_true=expected,
        #     y_pred=predictions)
        # recall = recall_score(
        #     y_true=expected,
        #     y_pred=predictions)
        # f1 = f1_score(
        #     y_true=expected,
        #     y_pred=predictions)
        # l, m, n, support = precision_recall_fscore_support(
        #     y_true=expected,
        #     y_pred=predictions)


        return (f"--- Parameters --- "
                f"\nLearning rate: {self.learningRate}"
                f"Activation: \n{self.hidden_activation}"
                f"\nNumber of Hidden Layers: 2"
                f"\nNumber of Neurons per Hidden Layer: {self.hidden_layer_size}"
                f"\nFinal output weights: {self.weightsThree}"
                f"\n\n--- Results --- "
                f"\nAccuracy: {accuracy}")
        # f"\nPrecision: {precision}"
        # f"\nRecall: {recall}"
        # f"\nF1 Score: {f1}"
        # f"\nSupport: {support}")

    def run(self, xarray, yarray, training=True):
        self.clear()
        start = time.time()
        i = 0
        for (data, val) in zip(xarray, yarray):
            self.iterate(data, val, training)
            i += 1
            if i % 100 == 0:
                sys.stdout.write(
                    "Progress: %d%%  Running accuracy: %d%% \r"
                    % ((i / len(data)),
                       (1 - (self.wrong / (self.wrong + self.correct)))))

        print(f"Time elapsed: {time.time() - start}")

        return self.get_results()


    def iterate(self, x: np.ndarray, y: np.ndarray, train=True):

        x = x.flatten()
        self.data = x
        wOne = self.weightsOne
        wTwo = self.weightsTwo
        wThree = self.weightsThree

        bOne = self.bOne
        bTwo = self.bTwo
        bThree = self.bThree

        LOneOut = np.dot(x, wOne) / self.input_size

        # lTwo is a hidden layer that feeds everything through a tanh activation function before applying weights
        # and sending the results to LThree
        LTwo = self.hidden_activation(LOneOut)
        LTwoOut = np.dot(LTwo, wTwo) / self.hidden_layer_size

        LThree = self.output_activation(LTwoOut)
        LThreeOut = np.dot(LThree, wThree) / self.hidden_layer_size

        if np.argmax(LThreeOut) == np.argmax(y):
            self.correct += 1
        else:
            self.wrong += 1

        result = [0] * self.output_size
        result[np.argmax(LThreeOut)] = 1
        result = np.asarray(result)
        y = np.asarray(y)
        self.results.append(result)
        self.expected.append(y)

        if train:
            self.backprop(LOneOut, LTwoOut, LThreeOut, y, result)

    def backprop(self, LOneOut, LTwoOut, LThreeOut, expected, results):

        wOne = self.weightsOne
        wTwo = self.weightsTwo
        wThree = self.weightsThree

        bOne = self.bOne
        bTwo = self.bTwo
        bThree = self.bThree
        bFour = self.bThree

        prime1 = self.hidden_derivative(LOneOut).flatten()
        prime2 = self.hidden_derivative(LTwoOut).flatten()
        prime3 = self.output_derivative(LThreeOut).flatten()

        # Calculating the deltas (The change in weight values relative to the input values)
        # The first step of this is to calculate the overall error.

        L3error = -(expected - results)
        L3delta = L3error * prime3
        L2delta = np.dot(L3delta, wThree.T) * prime2
        L1delta = np.dot(L2delta, wTwo.T) * prime1

        # Updating second hidden layer weights.
        # We add an additional axis to LayerTwo and L3delta to enable the dot product.
        # Then we subtract the resulting matrix of the dot product from the bottom layer of weights.
        # Finally we round the resulting weights to 7 digits to keep things from exploding.

        LTwoOut = LTwoOut[np.newaxis]
        L3delta = L3delta[np.newaxis]
        wThree -= np.dot(LTwoOut.T, L3delta * self.learningRate)
        wThree = np.round(wThree, 7)

        # Updating first hidden layer weights.
        # We add an additional axis to LayerOne and L2delta to enable the dot product.
        # Then we subtract the resulting matrix of the dot product from the bottom layer of weights.
        # Finally we round the resulting weights to 7 digits to keep things from exploding.

        LOneOut = LOneOut[np.newaxis]
        L2delta = L2delta[np.newaxis]
        wTwo -= np.dot(LOneOut.T, L2delta * self.learningRate)
        wTwo = np.round(wTwo, 7)

        # Updating input layer weights.
        # We add an additional axis to the input data and L1delta to enable the dot product.
        # Then we subtract the resulting matrix of the dot product from the bottom layer of weights.
        # Finally we round the resulting weights to 7 digits to keep things from exploding.

        data = self.data[np.newaxis]
        L1delta = L1delta[np.newaxis]
        wOne -= np.dot(data.T, L1delta * self.learningRate)
        wOne = np.round(wOne, 7)

        # Then, we set the stored weights equal to our newly adjusted weights and increment the

        self.weightsThree = wThree
        self.weightsTwo = wTwo
        self.weightsOne = wOne

    # Reset results for this learner
    def _clear(self):
        self.correct = 0
        self.wrong = 0

        self.results = []
        self.expected = []





if __name__ == "__main__":

    train_x, train_y, test_x, test_y = load_data()

    # learner = OriginalLearner(
    #     hidden_activation="leaky_relu",
    #     output_activation="sigmoid",
    #     input_size=36100,
    #     learning_rate="neg_linear_learn_rate",
    #     output_size=4,
    #     hidden_layer_size=16)

    # (train_x, train_y), (test_x, test_y) = mnist.load_data()
    #
    # train_y_arr = []
    # for val in train_y:
    #     newVec = [0] * 10
    #     newVec[int(val)] = 1
    #     train_y_arr.append(newVec)
    # train_y = train_y_arr
    #
    # test_y_arr = []
    # for val in test_y:
    #     newVec = [0] * 10
    #     newVec[val] = 1
    #     test_y_arr.append(newVec)
    # test_y = test_y_arr

    # normalize values between 0 and 1
    # train_x = np.array([np.divide(sample,255).flatten() for sample in train_x])
    # test_x = np.array([np.divide(sample,255) for sample in test_x])

    num_samples, dimension_x = train_x.shape

    n_inputs = dimension_x

    # learner = OriginalLearner(
    #     hidden_activation="relu",
    #     output_activation="sigmoid",
    #     input_size=n_inputs,
    #     learning_rate=0.01,
    #     output_size=10,
    #     hidden_layer_size=32)

    perceptron = NeuralNetwork("constant", 0.01)

    perceptron.add_layer("relu", 16, 36100)
    perceptron.add_layer("relu", 16)
    perceptron.add_layer("sigmoid", 4)

    print("Beginning training iterations.")
    print(perceptron.run(train_x, train_y, training=True))
    print("Beginning testing iterations.")
    print(perceptron.run(train_x, train_y, training=False))
