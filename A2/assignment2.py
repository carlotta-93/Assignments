import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold  # create indices for CV
from sklearn import preprocessing  # version 1

# load of data
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split input variables and labels
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]

# EXERCISE 1
n_neighbors = 1
# TODO say why I use distance instead of uniform
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(XTrain, YTrain)

# compute the accuracy on the test set and on the train set
accTest = accuracy_score(YTest, clf.predict(XTest))
accTrain = accuracy_score(YTrain, clf.predict(XTrain))
print 'The accuracy score on the test set is %.10f and the accuracy score on the train set is %.10f' % (accTest, accTrain)


# PLOT THE DATA
def dimension_reduction(data_matrix):
    """This function reduces the dimensionality of the data matrix """
    pca = decomposition.PCA(n_components=3)
    red_matrix = pca.fit_transform(data_matrix)
    return red_matrix

fig = plt.figure()
ax = Axes3D(fig)
reduced_matrix = dimension_reduction(XTrain)
ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], reduced_matrix[:, 2],
           c=['#9999ff' if y == 1 else '#f2983e' for y in YTrain], edgecolor='none')
# plt.show()

# EXERCISE 2
k_list = [i for i in range(0, 26) if i % 2 != 0]
k_best_val = 0
c_error_val = 1


def perform_cross_validation(klist, kval, cval, x_train, y_train):
    """perform cross validation, returns best k value and it's classification error"""
    for k in klist:
        clf_cross_validation = KNeighborsClassifier(k, weights='distance')
        # hyperparameter selection using cross-validation
        cv_cross_validation = KFold(n_splits=5)
        # loop over CV folds
        for train, test in cv_cross_validation.split(x_train):
            XTrainCV, XTestCV, YTrainCV, YTestCV = x_train[train], x_train[test], y_train[train], y_train[test]
        # fit the model on the new sets
        clf_cross_validation.fit(XTrainCV, YTrainCV)
        # compute the classification error
        classification_error = zero_one_loss(YTestCV, clf_cross_validation.predict(XTestCV))
        if classification_error < cval:
            cval = classification_error
            kval = k
        print 'With %d clusters, the classification error is %.10f' % (k, classification_error)
        # using only 10 numbers after comma as these are the significant digits
    return kval, cval

k_best_found = perform_cross_validation(k_list, k_best_val, c_error_val, XTrain, YTrain)[0]
class_error_found = perform_cross_validation(k_list, k_best_val, c_error_val, XTrain, YTrain)[1]

print 'The best k is %d with classification error of %.10f' % (k_best_found, class_error_found)

# EXERCISE 3
# evaluation of classification performance
clf_k_best = KNeighborsClassifier(k_best_found, weights='distance')
clf_k_best.fit(XTrain, YTrain)
k_best_acc_test = accuracy_score(YTest, clf_k_best.predict(XTest))
k_best_acc_train = accuracy_score(YTrain, clf_k_best.predict(XTrain))
print 'The accuracy score of classifier with k_best on the test set is %.10f and on the train set is %.10f' \
      % (k_best_acc_test, k_best_acc_train)


# EXERCISE 4
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)

# perform cross validation
k_best_found_normalized = perform_cross_validation(k_list, k_best_val, c_error_val, XTrainN, YTrain)[0]
class_error_found_normalized = perform_cross_validation(k_list, k_best_val, c_error_val, XTrainN, YTrain)[1]
print 'The best k found in cross validation procedure is %d with classification error of %.10f' \
      % (k_best_found_normalized, class_error_found_normalized)

# fit the model
clfN = KNeighborsClassifier(k_best_found_normalized, weights='distance')
clfN.fit(XTrainN, YTrain)

# compute the accuracy on the test set and on the train set
accTestN = accuracy_score(YTest, clfN.predict(XTestN))
accTrainN = accuracy_score(YTrain, clfN.predict(XTrainN))

print 'The accuracy score of classifier with k_best on the test set normalized is %.10f ' \
      'and the accuracy score on the train set normalized is %.10f' % (accTestN, accTrainN)


# def distance(a, b):
#     return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))
