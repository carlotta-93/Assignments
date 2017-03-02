from itertools import groupby
import numpy as np

# dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
#
# # split input variables and labels
# XTrain = dataTrain[:, : -1]
# YTrain = dataTrain[:, -1]
# XTest = dataTest[:, : -1]
# YTest = dataTest[:, -1]

N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y_train = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

k_list = [i for i in range(0, 26) if i % 2 != 0]
k_best_val = 0
c_error_val = 1

num_folds = 5
subset_size = len(N) / num_folds
for n in range(num_folds):
    x_test_cv = N[n * subset_size: n * subset_size + subset_size]
    x_train_cv = N[:n * subset_size] + N[(n + 1) * subset_size:]
    y_test_cv = y_train[n * subset_size:n * subset_size + subset_size]
    y_train_cv = y_train[:n * subset_size] + y_train[(n + 1) * subset_size:]
    print ' j: ' + str(n)
    print 'x_train set: ' + str(x_train_cv)
    print 'y_train set: ' + str(y_train_cv)
    print 'x_test set: ' + str(x_test_cv)
    print 'y_test set: ' + str(y_test_cv)

# for j in k_list:
#     subset_size = len(N) / 5
#     testing_this_round = N[j * subset_size:][:subset_size]
#     training_this_round = N[:j * subset_size] + N[(j + 1) * subset_size:]
#     print ' j: ' + str(j)
#     print 'train set: ' + str(training_this_round)
#     print 'test set: ' + str(testing_this_round)


