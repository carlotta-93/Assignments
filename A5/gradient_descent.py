import numpy as np
from sympy import *
from sympy.abc import x
import matplotlib.pyplot as plt


def compute_derivative(x_s):
    """Compute derivative of function given as argument"""
    grad_function = exp(-x_s / 2) + 10 * (x_s ** 2)
    yprime = grad_function.diff(x_s)
    return yprime, grad_function


# transform the sympy function in lambda function
compute_grad = lambdify(x, compute_derivative(x)[0])  # derivative of the original function
compute_value = lambdify(x, compute_derivative(x)[1])  # original function

learning_rates = [0.1, 0.01, 0.001, 0.0001]


def find_grad_points(x_point, learning_rate, max_n_iteration):
    # Setting parameters for convergence check
    num_iter = 0  # This is the variable that will keep track of the number of iterations
    convergence = 0  # This is the variable that will keep track of whether we have converged
    points_found = []
    tolerance = 10**(-10)
    while convergence == 0:
        # Compute gradient and take a step in its direction
        grad = compute_grad(x_point)  # Computing gradient of objective function in current point
        points_found.append(x_point)  # Recording objective function value

        # Take a step in the direction of steepest ascent
        v_t = (-grad)
        new_x = x_point + learning_rate * v_t  # compute new point

        # Checking for convergence
        num_iter += 1
        points_distance = abs(x_point - new_x)
        x_point = new_x
        if points_distance < tolerance:
            convergence = 1
        elif num_iter > max_n_iteration:
            convergence = 1
    return points_found, num_iter

# points to compute the original function
x_p = np.arange(-4, 4, 0.01)
y_p = [compute_value(el) for el in x_p]

# starting point
x = 1
max_iter3 = 3
for l_rates in learning_rates:
    points, _ = find_grad_points(x, l_rates, max_iter3)
    print 'with learning rate: %f the points are %s' % (l_rates, points)
    y_values = [compute_value(el) for el in points]
    print 'the y values are: %s' % y_values
    j = 0.0
    for i in range(len(points)):
        m = compute_grad(points[i])
        b = compute_value(points[i]) - (m * points[i])
        y_1 = (m * -5) + b
        y_2 = (m * 5) + b
        plt.plot([-5, 5], [y_1, y_2], label='tangent on point %d' % (i + 1))
        blues = plt.get_cmap('Greens')

        plt.scatter(points[i], y_values[i], c=blues(j), label='grad desc step # %d' % i, s=20)
        j += 0.25
    plt.plot(x_p, y_p, color='#9999ff', label='original function')
    plt.title('Learning rate value is: %f' % l_rates)
    plt.axis([-4, 4, -10, 50])
    plt.legend(loc='lower right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

max_iter10 = 10
for l_rates in learning_rates:
    points, _ = find_grad_points(x, l_rates, max_iter10)
    print 'with learning rate: %f the points are %s' % (l_rates, points)
    y_values = [compute_value(el) for el in points]
    print 'the y values are: %s' % y_values
    blues = plt.get_cmap('Greens')
    j = 0.0
    for i in range(len(points)):
        plt.scatter(points[i], y_values[i], c=blues(j), label='grad desc step # %d' % i, s=20, lw=0.5)
        j += 0.1
    plt.plot(x_p, y_p, color='#9999ff', label='original function')
    plt.title('Learning rate value is: %f' % l_rates)
    plt.axis([-4, 4, -10, 50])
    plt.legend(loc='lower right', scatterpoints=1)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


max_iter10000 = 10000
for l_rates in learning_rates[1:]:  # not using learning rate 0.1
    points, number_iterations = find_grad_points(x, l_rates, max_iter10000)
    print 'with learning rate: %f' % l_rates
    n = len(points)
    print 'the function value in the last iteration is: %s' % points[n-1]
    print 'the final iteration is iteration number: %f' % (number_iterations-1)
