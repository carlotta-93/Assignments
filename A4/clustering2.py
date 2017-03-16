import numpy as np
import matplotlib.pyplot as plt

# read in the data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]

num_k = 2
max_iteration = 300


def pca(data_m):
    """This functions takes as argument the data loaded from the file and return the eigenvalues,
    the eigenvectors and the centered data set"""
    # center the matrix -> have zero mean
    data_mean = np.mean(data_m, axis=0)
    data_centered = data_m - data_mean

    # compute covariance
    cov_matrix = np.cov(data_centered.T)
    # compute eigenvectors and eigenvalues of symmetric matrix
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    # sort eigenvalues and the eigenvectors in the same order
    index_eval = np.argsort(eig_values)[::-1]  # reverts the order of the indexes
    eig_vectors = eig_vectors[:, index_eval]

    # sort eigenvectors according to same index
    eig_values = eig_values[index_eval]

    return eig_values, eig_vectors, data_centered


def mds(data_m, d):
    """This function takes as argument the data set, performs pca and returns the data
    points projected on the first d principal components """
    eig_vals, eig_vect, centered_data = pca(data_m)
    transformed_data = np.dot(eig_vect.T, centered_data.T).T
    return [transformed_data[:, n] for n in range(d)]

projected_data = mds(XTrain, 2)


def k_means(data, k, max_iter):
    """This function takes as input a data set, k=num_cluster, maximum value of iterations, performs
    the k-means algorithm and return the k centroids found, in a dictionary"""

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
centoids_projected = mds(centroids_found.values(), 2)


for i in range(len(projected_data[0])):
    if YTrain[i] == 0:
        c0 = plt.scatter(projected_data[0][i], projected_data[1][i], marker='o', c='orange', lw=0.3)
    else:
        c1 = plt.scatter(projected_data[0][i], projected_data[1][i], marker='o', c='violet', lw=0.3)
    centers = plt.scatter(centoids_projected[0], centoids_projected[1], marker='D', c='green', s=50, lw=0.2)
plt.axis('equal')
plt.xlabel('first PC')
plt.ylabel('second PC')
plt.title('Projection on first two PCs of weed crop data')
plt.legend([c0, c1, centers], ['points in class 0', 'points in class 1', 'centroids'], loc='lower right', scatterpoints=1)
plt.show()


