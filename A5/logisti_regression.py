import numpy as np
import matplotlib.pyplot as plt

train_2D1 = np.loadtxt('Iris2D1_train.txt')
test_2D1 = np.loadtxt('Iris2D1_test.txt')
# separate quality score from features
XTrain2D1 = train_2D1[:, :-1]  # points
YTrain2D1 = train_2D1[:, -1]  # labels
XTest2D1 = test_2D1[:, :-1]  # points
YTest2D1 = test_2D1[:, -1]  # labels

train_2D2 = np.loadtxt('Iris2D2_train.txt')
test_2D2 = np.loadtxt('Iris2D2_test.txt')
# separate quality score from features
XTrain2D2 = train_2D2[:, :-1]
YTrain2D2 = train_2D2[:, -1]
XTest2D2 = test_2D2[:, :-1]
YTest2D2 = test_2D2[:, -1]


for i in range(len(XTrain2D1)):
    if YTrain2D1[i] == 0:
        c0_tr = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', c='orange', lw=0.3)
    else:
        c1_tr = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('train data for dataset Iris2D1')
plt.legend([c0_tr, c1_tr], ['sepal length train set', 'petal length train set'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()

for i in range(len(XTest2D1)):
    if YTest2D1[i] == 0:
        c0_test = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', c='orange', lw=0.3)
    else:
        c1_test = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test data for dataset Iris2D1')
plt.legend([c0_test, c1_test], ['sepal length test set', 'petal length test set'], loc='lower right', scatterpoints=3)
plt.grid()

plt.show()
