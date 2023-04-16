# Lasso regression comparison implementation
# Copied from https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/
# Error changed to mean squared error

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
data_train, data_val = train_test_split(load_breast_cancer(), test_size = 0.2, random_state = 2)


Y_train = data_train.iloc[:, -1].values
X_train = data_train.iloc[:,0 : -1].values

X_test = data_val.iloc[:,0 : -1].values
def score(y_pred, y_true):

    error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
    score = 1 - error
    return score


actual_cost = list(data_val['COST'])
actual_cost = np.asarray(actual_cost)




from sklearn.linear_model import Lasso
#Initializing the Lasso Regressor with Normalization Factor as True
lasso_reg = Lasso(normalize=True)
#Fitting the Training data to the Lasso regressor
lasso_reg.fit(X_train,Y_train)
#Predicting for X_test
y_pred_lass =lasso_reg.predict(X_test)
#Printing the Score with RMLSE
print("\n\nLasso SCORE : ", score(y_pred_lass, actual_cost))
