import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split input variables and labels
XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]

rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf = rf_clf.fit(XTrain, YTrain)
predictions = rf_clf.predict(XTest)
accuracy_score = accuracy_score(YTest, predictions)

print 'The accuracy score for the test set is: %f' % accuracy_score
