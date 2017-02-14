# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
import numpy as np
import scipy.stats
data_matrix = np.loadtxt('smoking.txt')


def hyp_test(data):
    non_smokers = data[data[:, 4] == 0]
    smokers = data[data[:, 4] == 1]
    significance_level = 0.05
    result_values = scipy.stats.ttest_ind(smokers[:, 1], non_smokers[:, 1], equal_var=True)
    print "The p value of the samples is: " + str(result_values[1])
    if result_values[1] < significance_level:
        return True  # null hypothesis is rejected
    else:
        return False


print hyp_test(data_matrix)