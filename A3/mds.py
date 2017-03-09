# input:   1) datamatrix as loaded by numpy.loadtxt('dataset.txt')
# 2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
# output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top
# d PCs

import pca
import numpy as np
import matplotlib.pyplot as plt

pesticide_data = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
XTrain = pesticide_data[:, : -1]
num_dim = 2


def mds(data, d):
    """This function takes as argument the data set, performs pca and returns the data
    points projected on the first d principal components """
    pesticide_eig_vals, pesticide_eig_vect, pesticide_centered_data = pca.pca(data)
    transformed_data = np.dot(pesticide_eig_vect.T, pesticide_centered_data.T).T
    return [transformed_data[:, n] for n in range(d)]


def main():

    transformed = mds(XTrain, num_dim)
    plt.scatter(transformed[0], transformed[1], marker='o', c='orange', label='data points')
    plt.grid()
    plt.axis('equal')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Pesticide data projected on 2 principal components')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()