import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('diatoms.txt')


def plot_cells(cells, title):
    """This function takes as argument an array of 5 cells and plots them with 5 different color"""
    blues = plt.get_cmap('Greens')
    i = 0.0
    for cell in cells:
        x = cell[::2]
        y = cell[1::2]
        plt.scatter(x, y, c=blues(i), label='diatom')
        i += 0.2
    plt.axis('equal')
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.title(title)
    plt.legend(loc='lower right', scatterpoints=1)
    plt.show()


def main():
    # EXERCISE 1
    third_diatoms_coordinates = data[2]
    td_x = third_diatoms_coordinates[::2]
    td_y = third_diatoms_coordinates[1::2]

    all_x = []
    all_y = []
    for diatoms in data:
        all_x.append(diatoms[::2])
        all_y.append(diatoms[1::2])

    all_diatoms = plt.scatter(all_x, all_y, marker='o', c='#9999ff', label='all diatoms', s=10, facecolor='0.5', lw=0.5)
    third_d = plt.scatter(td_x, td_y, marker='o', c='orange', label='third diatoms', s=10, facecolor='0.5', lw=0.5)
    plt.axis('equal')
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.title('Shape of all diatoms')
    plt.legend(loc='lower right')
    plt.show(all_diatoms)
    plt.show(third_d)

    # EXERCISE 2
    data_mean = np.mean(data, axis=0)
    Sigma = np.cov(data.T)
    # compute eigenvectors and eigenvalues
    eigen_values, eigen_vect = np.linalg.eig(Sigma)

    # compute standard deviations
    s0 = np.sqrt(eigen_values[0])
    e0 = eigen_vect[:, 0]

    s1 = np.sqrt(eigen_values[1])
    e1 = eigen_vect[:, 1]

    s2 = np.sqrt(eigen_values[2])
    e2 = eigen_vect[:, 2]

    cells_along_pc1 = np.zeros((5, 180))
    cells_along_pc2 = np.zeros((5, 180))
    cells_along_pc3 = np.zeros((5, 180))

    # visualize the movement along the first PC
    for i in range(5):
        cells_along_pc1[i, :] = data_mean + (i - 2) * s0 * e0
    plot_cells(cells_along_pc1, title='Variance along the first PC')
    # visualize the movement along the second PC
    for i in range(5):
        cells_along_pc2[i, :] = data_mean + (i - 2) * s1 * e1
    plot_cells(cells_along_pc2, title='Variance along the second PC')

    # visualize the movement along the third PC
    for i in range(5):
        cells_along_pc3[i, :] = data_mean + (i - 2) * s2 * e2
    plot_cells(cells_along_pc3, title='Variance along the third PC')



    # plt.axis('equal')
    # plt.xlabel('x coordinates')
    # plt.ylabel('y coordinates')
    # plt.title('Shape of all diatoms')
    # plt.legend(loc='lower right')
    # plt.show(first_pt)





if __name__ == '__main__':
    main()
