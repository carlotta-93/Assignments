import numpy as np

# input: 1) f: the predicted values of the dependent output variable, an N-dimensional vector, as a numpy array
#        2) t: the ground truth values of dependent output variable, an N-dimensional vector, as a numpy array
#
# output: 1) the root mean square error (rmse) as a 1 x 1 numpy array


def rmse(f, t):
    sum_error = 0.0
    for i in range(len(t)):
        prediction_error = f[i] - t[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(t))
    return np.sqrt(mean_error)
