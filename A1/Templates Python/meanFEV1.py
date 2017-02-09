# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers

import numpy as np
import matplotlib.pyplot as plt
data_matrix = np.loadtxt('smoking.txt')


def mean_fev_one(data):
    non_smokers = data[data[:, 4] == 0]
    smokers = data[data[:, 4] == 1]
    avg_smokers = np.average(smokers[:, 1])
    avg_non_smokers = np.average(non_smokers[:, 1])
    plt.boxplot([smokers[:, 1], non_smokers[:, 1]])
    plt.setp(plt.gca(), xticks=[1, 2], xticklabels=['Smokers FEV1', 'Non_smokers FEV1'])
    plt.show()
    return avg_smokers, avg_non_smokers

print "The average FEV1 values for smoker and non smoker are: " + str(mean_fev_one(data_matrix))

