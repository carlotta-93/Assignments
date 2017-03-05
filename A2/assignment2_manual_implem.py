import numpy as np
import operator
from itertools import groupby

# load of data
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split input variables and labels
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]

k_list = [i for i in range(0, 26) if i % 2 != 0]
k_best_val = 0
c_error_val = 1
num_folds = 5


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
        # print label_list
        predictions.append(max(label_list_predicted, key=label_list_predicted.count))
    return predictions


def get_accuracy(test_label, predictions, y_test):
    """this function get the accuracy score of the predicted labels comparing it with the values in the YTest set"""
    correct = 0
    for x in range(len(test_label)):
        if y_test[test_label[x][0][0]] == predictions[x]:
            correct += 1
    return correct / float(len(test_label))

# EXERCISE 2


def cross_validation(n_splits, x_train, y_train, klist, kval, cval):
    """this function perform cross validation and returns the best k value found with its classification error"""
    subset_size = len(x_train) / n_splits
    for k in klist:
        # split the data
        l = []
        for n in range(num_folds):
            x_test_cv = x_train[n * subset_size: n * subset_size + subset_size]
            y_test_cv = y_train[n * subset_size:n * subset_size + subset_size]
            x_train_cv = np.concatenate((x_train[:n * subset_size], x_train[(n + 1) * subset_size:]))
            y_train_cv = np.concatenate((y_train[:n * subset_size], y_train[(n + 1) * subset_size:]))
            neighbors = get_neigh(x_train_cv, x_test_cv, k)
            predicted = predict(neighbors, y_train_cv)
            classification_error = 1.0 - get_accuracy(neighbors, predicted, y_test_cv)
            l.append(classification_error)
        class_err_avg = sum(l)/len(l)  # average error for each k
        if class_err_avg < cval:
            cval = class_err_avg
            kval = k
        print 'With %d clusters, the classification error is %.10f' % (k, class_err_avg)
    return kval, cval


def main():
    # EXERCISE 1
    neighbors = get_neigh(XTrain, XTest, 1)
    predictions_test = predict(neighbors, YTrain)
    accuracy_test = get_accuracy(neighbors, predictions_test, YTest)
    print 'The accuracy score of the classifier on the test set, with k=1 is: ' + str(accuracy_test)
    predictions_train = predict(get_neigh(XTrain, XTrain, 1), YTrain)
    accuracy_train = get_accuracy(neighbors, predictions_train, YTrain)
    print 'The accuracy score of the classifier on the train set, with k=1 is: ' + str(accuracy_train)

    k_best_found, class_error_found = cross_validation(num_folds, XTrain, YTrain, k_list, k_best_val, c_error_val)
    print 'The best k is %d with classification error of %.10f' % (k_best_found, class_error_found)

    # EXERCISE 3
    # compute the accuracy of the classifier using the k_best found with cross validation
    neighbors_k_best = get_neigh(XTrain, XTest, k_best_found)
    predictions_k_best = predict(neighbors_k_best, YTrain)
    accuracy_k_best = get_accuracy(neighbors_k_best, predictions_k_best, YTest)
    print 'The accuracy score of classifier on the test set, with k_best_found is: ' + str(accuracy_k_best)

    predictions_train_k_best = predict(get_neigh(XTrain, XTrain, k_best_found), YTrain)
    accuracy_train_k_best = get_accuracy(neighbors, predictions_train_k_best, YTrain)
    print 'The accuracy score of classifier on the train set, with k_best_found is: ' + str(accuracy_train_k_best)

    # EXERCISE 4 - normalization
    # compute mean and standard deviation of train set
    mean_x_train = np.mean(XTrain, axis=0)
    standard_dev_x_train = np.std(XTrain, axis=0)

    # normalize data
    normalized_x_train = XTrain - mean_x_train / standard_dev_x_train
    normalized_x_test = XTest - mean_x_train / standard_dev_x_train

    k_best_found_norm, class_error_found_norm = cross_validation(num_folds, normalized_x_train, YTrain, k_list, k_best_val, c_error_val)
    print 'Normalized data -> The best k is %d with classification error of %.10f' % (k_best_found_norm, class_error_found_norm)

    neighbors_k_best_n = get_neigh(normalized_x_train, normalized_x_test, k_best_found_norm)
    predictions_k_best_n = predict(neighbors_k_best_n, YTrain)
    accuracy_k_best_n = get_accuracy(neighbors_k_best_n, predictions_k_best_n, YTest)
    print 'Normalized data -> The accuracy score of classifier on the test set normalized data, ' \
          'with k_best_found_norm is: ' + str(accuracy_k_best_n)

    # fit the model
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    clfN = KNeighborsClassifier(k_best_found_norm, weights='distance')
    clfN.fit(normalized_x_train, YTrain)
    # compute the accuracy on the test set and on the train set
    accTestN = accuracy_score(YTest, clfN.predict(normalized_x_test))
    print 'The accuracy score of classifier on normalized datafound with built-in functions is: ' + str(accTestN)


if __name__ == "__main__":
    main()
