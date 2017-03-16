# input: datamatrix as loaded by numpy.loadtxt('dataset.txt') output:  1) the eigenvalues in a vector (numpy array)
# in descending order 2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in
# the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the
# same order as their associated eigenvalues

import numpy as np
import matplotlib.pyplot as plt

murder_data = np.loadtxt('murderdata2d.txt')
pesticide_data = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
XTrain = pesticide_data[:, : -1]


def pca(data):
    """This functions takes as argument the data loaded from the file and return the eigenvalues,
    the eigenvectors and the centered data set"""
    # center the matrix -> have zero mean
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean

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



def main():
    """This function performs the pca on the two dataset and plot the results"""
    # EXERCISE 1.2
    murder_eig_values, murder_eig_vectors, centered_data = pca(murder_data)

    murder_feature1 = centered_data[:, 0]
    murder_feature2 = centered_data[:, 1]
    mean_cen = np.mean(centered_data, axis=0)

    s0 = np.sqrt(murder_eig_values[0])
    s1 = np.sqrt(murder_eig_values[1])

    # plot data
    plt.scatter(mean_cen[0], mean_cen[1], marker='*', c='r', label='mean of the data')
    plt.scatter(murder_feature1, murder_feature2, marker='o', c='orange', label='murder_data points')
    plt.plot([0, s0*murder_eig_vectors[0, 0]], [0, s0*murder_eig_vectors[0, 1]], 'b', linewidth=1.3,
             label='eigenvectors', c='purple')
    plt.plot([0, s1*murder_eig_vectors[1, 0]], [0, s1*murder_eig_vectors[1, 1]], 'b', linewidth=1.3, c='purple')

    plt.grid()
    plt.axis('equal')
    plt.xlabel('percentage of unemployed')
    plt.ylabel('murders per year per 1000000 inhabitants')
    plt.title('Murder data eigenvectors on centered data')
    plt.legend(loc='lower right')
    plt.show()

    # EXERCISE 1.3
    pesticide_eig_vals, pesticide_eig_vect, pesticide_centered_data = pca(XTrain)
    # plot data
    pc_index = plt.plot(pesticide_eig_vals, linewidth=2.0, label='variance', c='orange')
    plt.xticks(np.arange(0, 14))
    plt.grid()
    plt.xlabel('PC index')
    plt.ylabel('Variance')
    plt.title('Variance against PC index')
    plt.legend(loc='upper right')
    plt.show(pc_index)

    # Cumulative normalized variance
    cumulative_var = np.cumsum(pesticide_eig_vals / np.sum(pesticide_eig_vals))
    cum_var = plt.plot(cumulative_var, linewidth=2.0, label='cumulative variance', c='orange')
    plt.xticks(np.arange(0, 14))
    plt.grid()
    plt.xlabel('PC index')
    plt.ylabel('Cumulative variance')
    plt.title('Cumulative variance against PC index')
    plt.legend(loc='upper right')
    plt.show(cum_var)


if __name__ == "__main__":
    main()
