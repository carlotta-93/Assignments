import numpy as np

# read in the data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
XTrain = dataTrain[:, :-1]

num_k = 2
max_iteration = 300


def k_means(data, k, max_iter):
    """This function takes as input a data set, k=num_cluster, maximum value of iterations, performs
    the k-means algorithm and return the k centroids found"""

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

centroids_found = k_means(XTrain, num_k, max_iteration)

for centroid in centroids_found:
    print centroids_found[centroid]


