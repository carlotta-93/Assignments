# x and y should be vectors of equal length
# should return their correlation as a number

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

data_matrix = np.loadtxt('smoking.txt')
data = data_matrix[np.argsort(data_matrix[:, 0])]

ages = data[:, 0]
fev1_values = data[:, 1]

def corr(x, y):
    correlation_value = scipy.stats.pearsonr(x, y)
    plt.plot(x, y, 'bo', color='darkgreen')
    plt.ylabel('FEV1 values')
    plt.xlabel('ages')
    plt.grid(True)
    plt.show()
    return correlation_value


print "The Pearson correlation coefficient, and the p-value are: " + str(corr(ages, fev1_values))


non_smokers = data[data[:, 4] == 0]
smokers = data[data[:, 4] == 1]

# n, bins, patches = plt.hist(non_smokers[:, 0], 30, alpha=0.9, facecolor='orange', label='age of non smokers')
# n, bins, patches = plt.hist(smokers[:, 0], 30, alpha=0.8, facecolor='purple', label='ages of smokers')

plt.xlabel('ages of smokers and non smokers')
plt.ylabel('Frequency')
plt.title('ages of smoker')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
