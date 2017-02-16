# x and y should be vectors of equal length
# should return their correlation as a number

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go

data_matrix = np.loadtxt('smoking.txt')
data = data_matrix[np.argsort(data_matrix[:, 0])]

ages = data[:, 0]
fev1_values = data[:, 1]


def corr(x, y):
    """This function takes as arguments the values of the age and their relative FEV1 values from the data and creates a
    scattered plot; it computes and returns the correlation value """

    correlation_value = scipy.stats.pearsonr(x, y)
    covariance_m = np.cov(x, y)
    std_x = np.std(x)
    std_y = np.std(y)
    correlation = covariance_m / (std_x * std_y)
    print 'The correlation coefficient manually computed is: ' + str(correlation[0][1])
    trace = go.Scatter(x=x, y=y, mode='markers',
                       marker=dict(size=7, color='rgba(153, 153, 255, .8)', line=dict(width=.4))
                       )
    layout = go.Layout(title='Fev1 values and ages',
                       xaxis=dict(title='ages', gridwidth=2),
                       yaxis=dict(title='FEV1 values', gridwidth=2),
                       plot_bgcolor='rgb(224, 224, 224)'
                       )
    fig = dict(data=[trace], layout=layout)
    py.offline.plot(fig)
    return correlation_value

print "The Pearson correlation coefficient computed with the built-in formula is : " + str(corr(ages, fev1_values)[0])

# histograms in exercise 5
non_smokers = data[data[:, 4] == 0]
smokers = data[data[:, 4] == 1]

n, bins, patches = plt.hist(non_smokers[:, 0], 30, alpha=0.9, facecolor='orange', label='age of non smokers')
n, bins, patches = plt.hist(smokers[:, 0], 30, alpha=0.8, facecolor='purple', label='ages of smokers')

plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('histogram over the age of subjects')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
