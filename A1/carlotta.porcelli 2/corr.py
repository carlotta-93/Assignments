# x and y should be vectors of equal length
# should return their correlation as a number

import numpy as np
import scipy.stats
import plotly as py
import plotly.graph_objs as go

data_matrix = np.loadtxt('smoking.txt')

# sort the data for age values
data = data_matrix[np.argsort(data_matrix[:, 0])]

ages = data[:, 0]
fev1_values = data[:, 1]


def corr(x, y):
    """This function takes as arguments the values of the age and their relative FEV1 values from the data and creates a
    scattered plot; it computes and returns the correlation value """

    # correlation coefficient using the built-in function
    correlation_value = scipy.stats.pearsonr(x, y)

    # correlation coefficient manually computed
    covariance_m = np.cov(x, y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    correlation = covariance_m / (std_x * std_y)
    print 'The correlation coefficient manually computed is: ' + str(correlation[0][1])

    return correlation_value


print "The Pearson correlation coefficient computed with the built-in formula is : " + str(corr(ages, fev1_values)[0])

# plot the data
non_smokers = data[data[:, 4] == 0]
smokers = data[data[:, 4] == 1]

sm_ages = smokers[:, 0]
sm_fev1 = smokers[:, 1]

non_sm_ages = non_smokers[:, 0]
non_sm_fev1 = non_smokers[:, 1]

trace_smokers = go.Scatter(x=sm_ages, y=sm_fev1, mode='markers',
                           marker=dict(size=7, color='rgba(153, 153, 255, .9)', line=dict(width=.4)),
                           name='smokers')

trace_non_smokers = go.Scatter(x=non_sm_ages, y=non_sm_fev1, mode='markers',
                               marker=dict(size=7, color='rgba(255, 178, 102, .8)', line=dict(width=.4)),
                               name='non smokers')

layout = go.Layout(title='Fev1 values over age of subjects for smokers and non-smokers',
                   xaxis=dict(title='ages', gridwidth=2),
                   yaxis=dict(title='FEV1 values', gridwidth=2),
                   plot_bgcolor='rgb(224, 224, 224)'
                   )
fig = go.Figure(data=[trace_non_smokers, trace_smokers], layout=layout)
py.offline.plot(fig)

# histograms in exercise 5
his_smoker = go.Histogram(x=smokers[:, 0],
                          marker=dict(color='rgb(153, 153, 255)'),
                          xbins=dict(size=1),
                          name='smokers')

his_non_smoker = go.Histogram(x=non_smokers[:, 0],
                              marker=dict(color='rgb(255, 178, 102)'),
                              xbins=dict(size=1),
                              name='non-smokers')

layout_his = go.Layout(title='histogram over the age of subjects',
                       xaxis=dict(title='ages', dtick=1),
                       yaxis=dict(title='frequency'),
                       bargap=0.3,
                       bargroupgap=0.3
                       )
data_overlap_hist = [his_smoker, his_non_smoker]
data_his_smoker = [his_smoker]
data_his_non_smoker = [his_non_smoker]
histograms = go.Figure(data=data_overlap_hist, layout=layout_his)
smoker_his = go.Figure(data=data_his_smoker, layout=layout_his)
non_smoker_his = go.Figure(data=data_his_non_smoker, layout=layout_his)

# uncomment these lines to make the plots
py.offline.plot(histograms)
# py.offline.plot(smoker_his)
# py.offline.plot(non_smoker_his)
