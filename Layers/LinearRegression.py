import matplotlib.pyplot as plt

import numpy as np

from Math.Functions import squared_error

import pandas as pd


class LinearRegression:

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.X = None
        self.Y = None
        self.output = None

    def fit(self, X: np.ndarray, Y: np.ndarray, regularization="L1", alpha=1.0, allow_negatives=False):
        # Fits the model

        # Initialize coefficients and store our data to examine
        self.coefficients = np.zeros(1)
        self.intercept = 0
        self.X = X
        self.Y = Y

        for x, y in zip(X, Y):
            prediction = self.forward_prop(x)

            regularize = 0
            if regularization == 'L1':
                regularize = np.sum(self.coefficients, axis=0) * alpha

            # Get our networkLoss
            # error = -(y - prediction)
            loss = -(y - prediction) ** 2 + regularize

            # Back propagation
            # Our derivative wrt our weights is to 2 * networkLoss * x
            # We take out the 2 because why not bro fr fr ong?
            # norm_term = float(2 / len(x))
            norm_term = 1
            x = x[np.newaxis]
            # regression_term = np.matmul(x.T, [error])
            regression_term = x * loss

            self.coefficients -= regression_term.flatten() * 0.1 * norm_term
            self.intercept -= loss * 0.1 * norm_term

        plt.plot(self.coefficients, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red',
                 label=r'Lasso; $\alpha = 1$', zorder=7)  # alpha here is for transparency

    def forward_prop(self, x):

        # self.output = (np.matmul(x, self.coefficients)) + self.intercept
        self.output = x * self.coefficients + self.intercept

        return self.output

    def predict(self, x):
        return self.forward_prop(x)


def single_val_regression(X, Y, epochs=20000, lrn_rate=0.0001):

    m = 0
    b = 0
    for i in range(epochs):
        for x, y in zip(X, Y):
            result = m * x + b
            error = y - result

            delta_m = -2 * x * (y - result)
            delta_b = -2 * error

            m -= delta_m * lrn_rate
            b -= delta_b * lrn_rate

    return m, b


if __name__ == "__main__":



    # data = pd.read_csv("DataHandling/Datasets/homicide.csv")
    #
    # train_x = data.age.values[5:]
    # train_y = data.num_homicide_deaths.values[5:]
    #
    # test_x = data.age.values[:5]
    # test_y = data.num_homicide_deaths.values[:5]
    #
    # model = LinearRegression()
    # model.fit(train_x, train_y, regularization='None')
    #
    # cost = 0
    # for x, y in zip(test_x, test_y):
    #     prediction = model.predict(x)
    #     cost += (y - prediction) ** 2
    #
    # avg_cost = cost / len(test_x)
    #
    # print(f"Average cost: {avg_cost}")
    #
    # data = pd.read_csv("DataHandling/Datasets/homicide.csv")
    # X = data.age.values[5:]
    # Y = data.num_homicide_deaths.values[5:]
    #
    # test_x = data.age.values[:5]
    # test_y = data.num_homicide_deaths.values[:5]
    #
    # m, b = single_val_regression(X, Y)
    #
    # networkLoss = 0
    # for x, y in zip(test_x, test_y):
    #     prediction = m * x + b
    #     networkLoss += (y - prediction) ** 2
    #
    # networkLoss = networkLoss / len(test_x)
    #
    # print(f"Loss: {networkLoss}")



