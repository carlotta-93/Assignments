import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('pca_toydata.txt')


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

pc_projected_data = mds(data, 2)
toy = plt.scatter(pc_projected_data[0], pc_projected_data[1], marker='o', c='orange', label='toy data', s=30, facecolor='0.5')
plt.axis('equal')
plt.xlabel('first PC')
plt.ylabel('second PC')
plt.title('Projection on first two PCs of toy data')
plt.legend(loc='lower right')
plt.show(toy)

proj_data_2 = mds(data[0:len(data)-2], 2)
toy_minus_two = plt.scatter(proj_data_2[0], proj_data_2[1], marker='o', c='orange', label='toy data', s=30, facecolor='0.5')
plt.axis('equal')
plt.xlabel('first PC')
plt.ylabel('second PC')
plt.title('Projection on first two PCs of toy data without las two points')
plt.legend(loc='lower right')
plt.show(toy_minus_two)
