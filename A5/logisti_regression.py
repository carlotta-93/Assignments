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

# PLOTS 2D1 dataset
for i in range(len(XTrain2D1)):
    if YTrain2D1[i] == 0:
        c0_tr_D1 = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_tr_D1 = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('Train data for dataset Iris2D1')
plt.legend([c0_tr_D1, c1_tr_D1], ['Iris setosa', 'Iris virginica'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()

for i in range(len(XTest2D1)):
    if YTest2D1[i] == 0:
        c0_test_D1 = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_test_D1 = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('Test data for dataset Iris2D1')
plt.legend([c0_test_D1, c1_test_D1], ['Iris setosa', 'Iris virginica'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()

# PLOTS 2D2 dataset
for i in range(len(XTrain2D2)):
    if YTrain2D2[i] == 0:
        c0_tr_D2 = plt.scatter(XTrain2D2[:, 0][i], XTrain2D2[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_tr_D2 = plt.scatter(XTrain2D2[:, 0][i], XTrain2D2[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Train data for dataset Iris2D2')
plt.legend([c0_tr_D2, c1_tr_D2], ['Iris virginica', 'Iris versicolor'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()

for i in range(len(XTest2D2)):
    if YTest2D2[i] == 0:
        c0_test_D2 = plt.scatter(XTest2D2[:, 0][i], XTest2D2[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_test_D2 = plt.scatter(XTest2D2[:, 0][i], XTest2D2[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Test data for dataset Iris2D2')
plt.legend([c0_test_D2, c1_test_D2], ['Iris virginica', 'Iris versicolor'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()


def sigmoid(s):
    """This is the logistic function"""
    return np.exp(s) / (1 + np.exp(s))


def in_sample_error(x_set, y_set, weigh):
    N = len(x_set)
    E = 0.0
    # print(y_set[0] * np.dot(weights, x_test[0, :]))
    for n in range(N):
        E += (1.0 / N) * np.log(1 + np.exp(np.dot(-y_set[n], np.dot(weigh, x_set[n]))))
    return E


# add 1s in the x_sets
x_train_2D1 = np.concatenate((np.ones((len(XTrain2D1), 1)), XTrain2D1), axis=1)
x_test_2D1 = np.concatenate((np.ones((len(XTest2D1), 1)), XTest2D1), axis=1)
# Turn the admission labels into +/- 1 labels
# y_train_2D1 = (YTrain2D1-0.5) * 2
y_test_2D1 = (YTest2D1 - 0.5) * 2

# add 1s in the x_sets
x_train_2D2 = np.concatenate((np.ones((len(XTrain2D2), 1)), XTrain2D2), axis=1)
x_test_2D2 = np.concatenate((np.ones((len(XTest2D2), 1)), XTest2D2), axis=1)
# Turn the admission labels into +/- 1 labels
y_train_2D2 = (YTrain2D2 - 0.5) * 2
y_test_2D2 = (YTest2D2 - 0.5) * 2


def logistic_gradient(x_set, y_set, weigh):
    N = len(x_set)
    g = 0.0 * weigh
    for n in range(N):
        g += ((-1.0 / N) * (y_set[n] * x_set[n])) * (sigmoid(-y_set[n] * np.dot(weigh, x_set[n])))
    return g


def perform_logistic_regression(x_set, y_set, max_iter, threshold):
    learning_rate = 0.01
    length, num_feat = x_set.shape

    x_set = np.concatenate((np.ones((len(x_set), 1)), x_set), axis=1)
    weight = 0.1 * np.random.randn(num_feat + 1)  # initialize the weights
    value = in_sample_error(x_set, y_set, weight)
    num_iter = 1
    convergence = 0

    E_in = []
    while convergence == 0:
        num_iter += 1
        grad = logistic_gradient(x_set, y_set, weight)
        v_t = -grad
        weight_new = weight + learning_rate * v_t

        # Compute in-sample error for new w
        current_value = in_sample_error(x_set, y_set, weight_new)

        if current_value < value:
            weight = weight_new
            value = current_value
            E_in.append(value)
            learning_rate *= 1.1
        else:
            learning_rate *= 0.9

        # Checking for convergence

        grad_norm = np.linalg.norm(grad)
        if grad_norm < threshold:
            convergence = 1
        elif num_iter > max_iter:
            convergence = 1

    return weight, E_in


def predict(x_set, weights):
    arg = np.exp(np.dot(weights, x_set.T))
    prob_i = arg / (1 + arg)

    predic_class = []
    prob_threshold = 0.5
    for i in prob_i:
        if i >= prob_threshold:
            predic_class.append(1)
        elif i <= prob_threshold:
            predic_class.append(0)
    predic_class = np.array(predic_class)
    return predic_class


maxim_iter = 100000
tolerance = 0.001
y_train_2D1 = (YTrain2D1 - 0.5) * 2

weights_2D1_train, E = perform_logistic_regression(XTrain2D1, y_train_2D1, maxim_iter, tolerance)
weights_2D2_train, E = perform_logistic_regression(XTrain2D2, y_train_2D2, maxim_iter, tolerance)

predicted_classes_2D1_train = predict(x_train_2D1, weights_2D1_train)
predicted_classes_2D1_test = predict(x_test_2D1, weights_2D1_train)
predicted_classes_2D2_train = predict(x_train_2D2, weights_2D2_train)
predicted_classes_2D2_test = predict(x_test_2D2, weights_2D2_train)


def compute_error_loss(predicted_classes, real_class):
    incorrect = 0.0
    for i in range(len(predicted_classes)):
        if predicted_classes[i] != real_class[i]:
            incorrect += 1.
    error_rate = (incorrect / len(real_class))
    return error_rate


print "The three parameters for data 2D1 are: " + str(weights_2D1_train)
print "The three parameters for data 2D2 are: " + str(weights_2D2_train)

print "The 0-1 error loss for train set 2D1 is: %f" % (compute_error_loss(predicted_classes_2D1_train, YTrain2D1))
print "The 0-1 error loss for test set 2D1 is: %f" % (compute_error_loss(predicted_classes_2D1_test, YTest2D1))

print "The 0-1 error loss for train set 2D2 is: %f" % (compute_error_loss(predicted_classes_2D2_train, YTrain2D2))
print "The 0-1 error loss for test set 2D2 is: %f" % (compute_error_loss(predicted_classes_2D2_test, YTest2D2))

w = np.array([-2.1, 0.99, -2.2])


# Draw line segment from line with equation
# w[0] + w[1]*x + w[2]*y = 0
# If w[2] != 0 (means line is not vertical) then rewrite as
# y = -x*(w[1]/w[2]) -w[0]/w[2]
# if w[2] = 0, rewrite it as
# x = -w[0]/w[1]
# One should not have both w[2] = w[1] = 0, it cannot
# represent a line!
# choose two values for x that suits you. They could be

# min and max x-values of your dataset

def logistic_line(x_test, w_vector):
    x1 = min(x_test[:, 0])
    x2 = max(x_test[:, 0])
    a = -w_vector[1] / w_vector[2]
    b = -w_vector[0] / w_vector[2]
    y1 = a * x1 + b
    y2 = a * x2 + b
    return x1, x2, y1, y2


# PLOTS 2D1 dataset
x1_trainD1, x2_trainD1, y1_trainD1, y2_trainD1 = logistic_line(XTrain2D1, weights_2D1_train)

for i in range(len(XTrain2D1)):
    if YTrain2D1[i] == 0:
        c0_tr_D1 = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_tr_D1 = plt.scatter(XTrain2D1[:, 0][i], XTrain2D1[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
D1_train, = plt.plot([x1_trainD1, x2_trainD1], [y1_trainD1, y2_trainD1], color='green')
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('Train data for dataset Iris2D1')
plt.legend([c0_tr_D1, c1_tr_D1, D1_train], ['Iris setosa','Iris virginica','logistic line'], loc='lower right', scatterpoints=3)
plt.grid()
plt.show()

x1_testD1, x2_testD1, y1_testD1, y2_testD1 = logistic_line(XTest2D1, weights_2D1_train)
for i in range(len(XTest2D1)):
    if YTest2D1[i] == 0:
        c0_test_D1 = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_test_D1 = plt.scatter(XTest2D1[:, 0][i], XTest2D1[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
D1_test, = plt.plot([x1_testD1, x2_testD1], [y1_testD1, y2_testD1], color='green')
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('Test data for dataset Iris2D1')
plt.legend([c0_test_D1, c1_test_D1, D1_test], ['Iris setosa', 'Iris virginica', 'logistic line'], loc='lower right',
           scatterpoints=3)
plt.grid()
plt.show()

# PLOTS 2D2 dataset

x1_trainD2, x2_trainD2, y1_trainD2, y2_trainD2 = logistic_line(XTrain2D2, weights_2D2_train)
for i in range(len(XTrain2D2)):
    if YTrain2D2[i] == 0:
        c0_tr_D2 = plt.scatter(XTrain2D2[:, 0][i], XTrain2D2[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_tr_D2 = plt.scatter(XTrain2D2[:, 0][i], XTrain2D2[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
D2_train, = plt.plot([x1_trainD2, x2_trainD2], [y1_trainD2, y2_trainD2], color='green')
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Train data for dataset Iris2D2')
plt.legend([c0_tr_D2, c1_tr_D2, D2_train], ['Iris virginica', 'Iris versicolor', 'logistic line'], loc='lower right',
           scatterpoints=3)
plt.grid()
plt.show()

x1_testD2, x2_testD2, y1_testD2, y2_testD2 = logistic_line(XTrain2D2, weights_2D2_train)
for i in range(len(XTest2D2)):
    if YTest2D2[i] == 0:
        c0_test_D2 = plt.scatter(XTest2D2[:, 0][i], XTest2D2[:, 1][i], marker='o', s=30, c='orange', lw=0.3)
    else:
        c1_test_D2 = plt.scatter(XTest2D2[:, 0][i], XTest2D2[:, 1][i], marker='o', s=30, c='#9999ff', lw=0.3)
dd, = plt.plot([x1_testD2, x2_testD2], [y1_testD2, y2_testD2], color='green')
plt.axis('tight')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Test data for dataset Iris2D2')
plt.legend([c0_test_D2, c1_test_D2, dd], ['Iris virginica', 'Iris versicolor', 'logistic line'], loc='lower right',
           scatterpoints=3)
plt.grid()
plt.show()
