# input: 1) X: the independent variables (data matrix), an (N x D)-dimensional matrix, as a numpy array
#        2) y: the dependent variable, an N-dimensional vector, as a numpy array
#
# output: 1) the regression coefficients, a (D+1)-dimensional vector, as a numpy array
#
# note: remember to either expect an initial column of 1's in the input X, or to append this within your code
import numpy as np
from rmse import rmse


data_train = np.loadtxt('redwine_training.txt')
data_test = np.loadtxt('redwine_testing.txt')

# separate quality score from features
XTrain = data_train[:, :-1]
YTrain = data_train[:, -1]
XTest = data_test[:, :-1]
YTest = data_test[:, -1]

# EXERCISE 1


def multivarlinreg(X, y):
    """This function computes the coefficient of the linear model"""
    x_matrix_transposed = np.transpose(X)
    pseudo_inverse = np.dot(np.linalg.inv(np.dot(x_matrix_transposed, X)), x_matrix_transposed)
    coefficients = np.dot(pseudo_inverse, y)
    return coefficients

# append list of ones to matrices
x_train_matrix = np.concatenate((np.ones((len(XTrain), 1)), XTrain), axis=1)
x_test_matrix = np.concatenate((np.ones((len(XTest), 1)), XTest), axis=1)

# compute coefficients for one feature and all features set
coefficients_first_feat = multivarlinreg(x_train_matrix[:, :2], YTrain)
coefficients_all_feat = multivarlinreg(x_train_matrix, YTrain)
print 'The weights for first-feature training set are:' + str(coefficients_first_feat)
print 'The weights for 12-features training set are:' + str(coefficients_all_feat)

# EXERCISE 2

predictions_one_feat = []
for points in x_test_matrix[:, :2]:
    y_hat = np.sum(np.dot(points, coefficients_first_feat))
    predictions_one_feat.append(y_hat)

predictions_all_feat = []
for points in x_test_matrix:
    y_hat_all_feat = np.sum(np.dot(points, coefficients_all_feat))
    predictions_all_feat.append(float(y_hat_all_feat))


print 'The rmse for one feature test set is: %f' % rmse(predictions_one_feat, YTest)
print 'The rmse for all features test set is: %f' % rmse(predictions_all_feat, YTest)


