import operator
from itertools import groupby
import numpy as np

dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split input variables and labels
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]

# N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# y_train = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0]


def get_neigh(train_set, test_set, k_n):
    """this function return the k-nearest neighbors"""
    distances = []
    for l in range(len(train_set)):
        for j in range(len(test_set)):
            distance = np.sqrt(np.sum(np.square(train_set[l] - test_set[j])))
            distances.append([j, l, distance])  # test_point, train_point, distance between points
    distances.sort(key=operator.itemgetter(0))
    sorted_list = [list(j) for z, j in groupby(distances, lambda x: x[0])]  # group by XTest points
    for elements in sorted_list:
        elements.sort(key=lambda x: int(x[2]))  # sort elements by distances
        neighbor = []  # list of k neighbors
        for elem in sorted_list:
            neighbor.append(elem[:k_n])
    return neighbor


def predict(neighbors, y_train):
    """this function predict the labels of the points in the neighbors list and return a list of
        prediction for every point in the test set"""
    predictions = []
    for n in range(len(neighbors)):
        label_list_predicted = [y_train[el[1]] for el in neighbors[n]]
        predictions.append(max(label_list_predicted, key=label_list_predicted.count)) # TODO somthing here
    print label_list_predicted
    return predictions


def get_accuracy(test_label, predictions, y_test):
    """this function get the accuracy score of the predictions against the values in the YTest set"""
    correct = 0
    for x in range(len(test_label)):
        if y_test[test_label[x][0][0]] == predictions[x]:
            correct += 1
    correction = correct / float(len(test_label))
    return correction


neighbors = get_neigh(XTrain, XTest, 3)
predictions_test = predict(neighbors, YTrain)
accuracy_test = get_accuracy(neighbors, predictions_test, YTest)
print 'The accuracy score of the classifier on the test set, with k=1 is: ' + str(accuracy_test)

# predictions_train = predict(get_neigh(XTrain, XTrain, 1), YTrain)
# accuracy_train = get_accuracy(neighbors, predictions_train, YTrain)
# print 'The accuracy score of the classifier on the train set, with k=1 is: ' + str(accuracy_train)
