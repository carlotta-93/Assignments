import numpy as np
import matplotlib.pyplot as plt
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

XTrain = dataTrain[:, :-1]

startingPoint = np.vstack((XTrain[0], XTrain[1]))
colors = 10*['g', 'r', 'c', 'b', 'k']

nk = 2
max_iteration = 300


def k_means(data, k, max_iter):
    centroids = {}
    for i in range(k):  # the first two centroids are the first two points of data
        centroids[i] = data[i]

    for i in range(max_iter):  # clear the classification because we change centroids
        classifications = {}

        for j in range(k):
            classifications[j] = []

        for elements in data:
            # calculate the distances - list of classified points in the data set
            distances = [np.linalg.norm(elements - centroids[cent]) for cent in centroids]
            classif = distances.index(min(distances))
            classifications[classif].append(elements)

        for classif in classifications:
            # take the avg for all of the values that are in th previous class & redefine the new centroids
            centroids[classif] = np.average(classifications[classif], axis=0)
    return centroids

centroids_found = k_means(XTrain, nk, max_iteration)

for centroid in centroids_found:
    print centroids_found[centroid]
    #plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', c='k', s=100, linewidths=5)

# for classification in clf.classifications:
#     color = colors[classification]
#     for featureset in clf.classifications[classification]:
#         plt.scatter(featureset[0], featureset[1], marker='x', c=color, s=50, linewidths=2)
# plt.show()
