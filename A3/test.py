import numpy as np
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

XTrain = dataTrain[:, :-1]

YTrain = dataTrain[:, -1]


cov_m = np.cov(XTrain)
print cov_m

data_mean = np.mean(XTrain, axis=0)
print '---> mean: ' + str(data_mean)

centered = XTrain - data_mean

cov_cent = np.cov(centered.T)
print '---> cov_centered: ' +str(cov_cent)

