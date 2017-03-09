import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import zero_one_loss

# read in the data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# split input variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, :-1]
YTest = dataTest[:, -1]

startingPoint = np.vstack((XTrain[0], XTrain[1]))
# kmeans = KMeans(n_init=1, init=startingPoint, n_clusters=2, algorithm='full', max_iter=300).fit(XTrain)

# print kmeans.cluster_centers_
# print startingPoint
k=2
n=300

print startingPoint
print startingPoint[1:]

def k_means(data, k, n):
    # initialize centers
    centers = np.vstack((data[0], data[1]))
    J=[]

    for iteration in range(n):
        for points in data:
            sqdistances = np.sum((centers-points)**2, axis=1)
            closest = np.argmin(sqdistances, axis=0)
            J.append(euclidean_dist(points, centers))

            for i in range(k):
                centers[i:] = points[closest:].mean(axis=0)
        J.append(euclidean_dist(points, centers))
    return centers


def euclidean_dist(points, centroids):
    """Calculates euclidean distance between a data point and the centroids"""
    distance = np.sqrt(np.sum(np.square(points - centroids)))
    return distance


# check if clusters have converged



print k_means(XTrain, k, n)