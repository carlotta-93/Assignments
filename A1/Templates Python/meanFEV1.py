# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
# load of data
data_matrix = np.loadtxt('smoking.txt')


def mean_fev_one(data):
    """This function takes a data matrix as arguments, divides it into two groups: smokers and non-smokers
    according to the values in the 5th column; it returns the average of the FEV1 values in the 2nd column
    of the subgroups and plot the results in a box plot """
    non_smokers = data[data[:, 4] == 0]
    smokers = data[data[:, 4] == 1]
    avg_smokers = np.mean(smokers[:, 1])
    avg_non_smokers = np.mean(non_smokers[:, 1])
    plt.boxplot([smokers[:, 1], non_smokers[:, 1]])
    plt.setp(plt.gca(), xticks=[1, 2], xticklabels=['Smokers', 'Non_smokers'])
    plt.ylabel('FEV1 values')
    plt.title('FEV1 average values for smokers and non-smokers')
    plt.show()
    trace0 = go.Box(y=smokers[:, 1], name='smokers')
    trace1 = go.Box(y=non_smokers[:, 1], name='non smokers')
    data = [trace0, trace1]
    layout = go.Layout(yaxis=dict(title='FEV1 values',zeroline=False))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

    return avg_smokers, avg_non_smokers

print "The average FEV1 values for smoker and non smoker are: " + str(mean_fev_one(data_matrix))

