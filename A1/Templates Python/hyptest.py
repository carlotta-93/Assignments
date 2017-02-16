# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
import numpy as np
import scipy.stats

data_matrix = np.loadtxt('smoking.txt')


def hyp_test(data):
    """This function takes as input the data matrix and returns a response indicating acceptance or rejection
    of the null hypothesis for a two-sample T-test """
    non_smokers = data[data[:, 4] == 0]
    smokers = data[data[:, 4] == 1]
    significance_level = 0.05
    # the two population variances are not assumed to be equal
    result_values = scipy.stats.ttest_ind(smokers[:, 1], non_smokers[:, 1], equal_var=False)
    # p value computed using the formula from the slides
    smoker_mean = np.mean(smokers[:, 1])
    non_smoker_mean = np.mean(non_smokers[:, 1])
    sv_smoker = np.var(smokers[:, 1], ddof=1)
    sv_nsmoker = np.var(non_smokers[:, 1], ddof=1)
    smk_len = len(smokers[:, 1])
    nsmoker_len = len(non_smokers[:, 1])
    t_value = (smoker_mean - non_smoker_mean) / np.sqrt((sv_smoker / smk_len) + (sv_nsmoker / nsmoker_len))
    print 'The t value computed manually is: ' + str(t_value)
    print 'The t value computed with the built in formula is: ' + str(result_values[0])
    degrees_of_freedom = np.floor((((sv_smoker / smk_len) + (sv_nsmoker / nsmoker_len)) ** 2) /
                                  ((sv_smoker ** 2 / (smk_len ** 2 * (smk_len - 1))) + (
                                      sv_nsmoker ** 2 / (nsmoker_len ** 2 * (nsmoker_len - 1)))))

    print 'Degrees of freedom: ' + str(degrees_of_freedom)
    p = 2 * scipy.stats.t.cdf(-t_value, degrees_of_freedom)
    print "The p value computed manually is: " + str(p)
    print "The p value computed with the built-in formula is: " + str(result_values[1])
    if result_values[1] < significance_level:
        return True  # null hypothesis is rejected
    else:
        return False


if hyp_test(data_matrix):
    print 'True - The null hypothesis is rejected'
else:
    print 'False - The null hypotesis is accepted'
