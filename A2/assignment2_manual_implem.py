import numpy as np
import operator
from collections import Counter
from sklearn.metrics import accuracy_score
from itertools import groupby

# load of data
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split input variables and labels
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]

k = 3


def get_neigh(train_set, test_set, k_n):
    distances = []
    for i in range(len(train_set)):
        for j in range(len(test_set)):
            distance = np.sqrt(np.sum(np.square(train_set[i] - test_set[j])))
            distances.append([j, i, distance]) # test_point, train_point, distance between points
    distances.sort(key=operator.itemgetter(0))
    sorted_list = [list(j) for i, j in groupby(distances, lambda x: x[0])]  # group by XTest points
    for elements in sorted_list:
        elements.sort(key=lambda x: int(x[2]))  # sort elements by distances
        neighbor = []  # list of k neighbors
        for elem in sorted_list:
            neighbor.append(elem[:k_n])
    return neighbor

tha_list = get_neigh(XTrain, XTest, k)[0:10]
for el in tha_list:
    print el
    for val in el:
        print val[1], YTrain[val[1]]

# def predict(neighbors):
#     for neighbours in neighbors:
#         return Counter(neighbours).most_common(1)

# for el in tha_list:
#     print el

# def getResponse(neighbors):
#     classVotes = {}
#     for x in range(len(neighbors)):
#         response = neighbors[x]
#         if response in classVotes:
#             classVotes[response] += 1
#         else:
#             classVotes[response] = 1
#     sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
#     return sortedVotes[0][0]
#
#
# def getAccuracy(testSet, predictions):
#     correct = 0
#     for x in range(len(testSet)):
#         if testSet[x] == predictions[x]:
#             correct += 1
#     return (correct / float(len(testSet))) * 100.0
#
# predictions=[]
#
#
# neighbors = get_neigh(XTrain, XTest, k)
# result = getResponse(neighbors)
# predictions.append(result)
# print('> predicted=' + repr(result) + ', actual=' + repr(YTest))
# accuracy = getAccuracy(YTest, predictions)
# print('Accuracy: ' + repr(accuracy) + '%')
# #
# #
