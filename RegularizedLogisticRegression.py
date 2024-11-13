import matplotlib.pyplot as plt
import numpy as np
import os


def get_point_marker_size():
    return 10


def get_marker_edge_width():
    return 2


def get_line_width():
    return 2


def get_font_size():
    return 16


def plot_objective_contours(X, y, lamb, w_min=-8, w_max=8, title=None, colors=None,
                            show_labels=True, new_figure=True, show_figure=True, save_filename=None):
    """
    Plots the logistic_regression.objective function with parameters
    X, y, lamb (lambda).

    X: Nx2 numpy ndarray, training input
    y: Nx1 numpy ndarray, training output
    lamb: Scalar lambda hyperparameter
    w_min (default=-8): Minimum of axes range
    w_max (default=8): Maximum of axes range
    title (default=None): Title of plot if not None
    colors (default=None): Color of contour lines. None will use default cmap.
    show_labels (default=True): Show numerical labels on contour lines
    new_figure (default=True): If true, calls plt.figure(), which create a
        figure. If false, it will modify an existing figure (if one exists).
    show_figure (default=True): If true, calls plt.show(), which will open
        a new window and block program execution until that window is closed
    save_filename (defalut=None): If not None, save figure to save_filename
    """
    N = 101

    w1 = np.linspace(w_min, w_max, N)
    w2 = np.linspace(w_min, w_max, N)
    W1, W2 = np.meshgrid(w1, w2)

    obj = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            w = np.array([[W1[i, j]], [W2[i, j]]])
            obj[i, j] = objective(w, X, y, lamb)

    # Ploting contour

    if new_figure:
        plt.figure(figsize=(8, 8))

    ax = plt.gca()
    contour_plot = ax.contour(W1, W2, obj, levels=20, colors=colors)
    if show_labels:
        ax.clabel(contour_plot, inline=1, fontsize=get_font_size())
    plt.tick_params(labelsize=get_font_size())
    # ax.set_xlabel('w1', fontsize = get_font_size())
    # ax.set_ylabel('w2', fontsize = get_font_size())

    ax.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    ax.set_axisbelow(True)

    if title is not None:
        plt.title(title)

    if save_filename is not None:
        plt.savefig(save_filename)

    if show_figure:
        plt.show()


def plot_optimization_path(point_list, color, linestyle='-', label=None):
    """
    Plot arrows stepping between points in the point list.

    point_list: List of 2D points, each of which is a 2x1 numpy ndarray
    color: matplotlib color
    linestyle: matplotlib linestyle
    label: Label to put in the plt.legend (plt.legend is not called in here)

    Does not call plt.figure() or plt.show()
    """
    X = []
    Y = []
    U = []
    V = []

    start = point_list[0]
    for point in point_list[1:]:
        X.append(start[0, 0])
        Y.append(start[1, 0])

        U.append(point[0, 0] - start[0, 0])
        V.append(point[1, 0] - start[1, 0])

        start = point

    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1,
               color=color, linestyle=linestyle, linewidth=get_line_width(), label=label)


def setup_data():
    """ Setup training dataset

        Returns tuple of length 2: (X_train, y_train)
        where X_train is an Nx2 ndarray and y_train is an Nx1 ndarry.
    """
    X_train = np.array([[0.0, 3.0], [1.0, 3.0], [0.0, 1.0], [1.0, 1.0]])
    y_train = np.array([[1], [1], [0], [0]])

    return (X_train, y_train)


def max_score():
    return 2


def timeout():
    return 60


def objective_test():
    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    X, y = setup_data()

    lambda_unit_test_values = [0, 0.1, 1, 10]
    w_unit_test_values = [np.array([[0.0], [0.0]]), np.array([[2.0], [2.0]]), np.array([[-2.0], [4.0]])]

    # Solution values
    expected_output_values = {}
    expected_output_values[0] = [2.773, 6.148, 6.145]
    expected_output_values[0.1] = [2.773, 6.548, 7.145]
    expected_output_values[1] = [2.773, 10.148, 16.145]
    expected_output_values[10] = [2.773, 46.148, 106.145]

    for lamb in lambda_unit_test_values:
        for i, w in enumerate(w_unit_test_values):
            actual_output = objective(w, X, y, lamb)
            if isinstance(actual_output, np.ndarray):
                actual_output = actual_output[0, 0]

            expected_output = expected_output_values[lamb][i]

            assert abs(
                actual_output - expected_output) < 0.01, 'Incorrect objective value found for lamda={}, w={}. Expected {}, found {}'.format(
                lamb, w, expected_output, actual_output)

        filename = '{}/objective_lambda_{:0.1f}.png'.format(figures_directory, lamb)
        filename = filename.replace('.', '_', 1)
        title = 'lambda = {}'.format(lamb)
        plot_objective_contours(X, y, lamb, title=title,
                                     new_figure=True, show_figure=False, save_filename=filename)

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output


def gradient_descent_test():
    figures_directory = 'figures'

    os.makedirs(figures_directory, exist_ok=True)

    X, y = setup_data()

    lamb = 0.1
    w0 = np.array([[7.0], [1.5]])
    num_iter = 10

    alpha_unit_test_values = [3, 0.9, 0.3]
    colors = ['steelblue', 'green', 'orange']

    # Solution values
    expected_w_lists = {}
    expected_w_lists[3] = [(7.000, 1.500), (1.901, -4.303), (4.081, 14.699), (-0.143, 4.289), (-3.053, -2.910),
                           (0.855, 15.799), (-2.402, 5.059), (-4.484, -2.243), (-0.143, 16.128), (-3.100, 5.289),
                           (-4.868, -1.980)]
    expected_w_lists[0.9] = [(7.000, 1.500), (5.470, -0.241), (4.090, 0.330), (2.839, -0.365), (1.887, 0.894),
                             (0.879, -0.472), (0.828, 2.561), (-0.117, 0.627), (-0.537, 0.175), (-0.405, 1.664),
                             (-1.060, 0.102)]
    expected_w_lists[0.3] = [(7.000, 1.500), (6.490, 0.920), (5.996, 0.431), (5.516, 0.131), (5.053, 0.034),
                             (4.605, 0.014), (4.172, 0.015), (3.756, 0.021), (3.357, 0.031), (2.975, 0.044),
                             (2.612, 0.062)]

    plot_objective_contours(X, y, lamb, title='Gradient Descent', colors='gray',
                                 show_labels=False, new_figure=True, show_figure=False, save_filename=None)

    for alpha, color in zip(alpha_unit_test_values, colors):
        actual_w_list = gradient_descent(X, y, lamb, alpha, w0, num_iter)

        expected_w_list = expected_w_lists[alpha]

        for i in range(num_iter + 1):
            assert abs(actual_w_list[i][0, 0] - expected_w_list[i][
                0]) < 0.01, 'Incorrect weight value found for iter={}, w[0]. Expected w={}, found w={}'.format(i,
                                                                                                               expected_w_list[
                                                                                                                   i],
                                                                                                               actual_w_list[
                                                                                                                   i])
            assert abs(actual_w_list[i][1, 0] - expected_w_list[i][
                1]) < 0.01, 'Incorrect weight value found for iter={}, w[1]. Expected w={}, found w={}'.format(i,
                                                                                                               expected_w_list[
                                                                                                                   i],
                                                                                                               actual_w_list[
                                                                                                                   i])

        plot_optimization_path(actual_w_list, color=color, label='alpha = {:.1f}'.format(alpha))

    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend(fontsize=get_font_size())

    filename = '{}/gradient_descent.png'.format(figures_directory)
    plt.savefig(filename)

    test_score = max_score()
    test_output = 'PASS\n'

    return test_score, test_output

###################################################
#   FEEL FREE TO WRITE ANY HELPER FUNCTIONS HERE  #
###################################################
def logistic(x,beta):
	y_pred = np.dot(x,beta)
	logistic_prob = 1/(1 + np.exp(-y_pred))
	return logistic_prob

def objective(beta, X, y, lamb):
    """ Given the parameters, w, the data, X and y, and the hyper parameter,
        lambda, return the objective function for L2-regularized logistic
        regression.

        Specifically, the objective function is the negative log likelihood
        plus the negative log prior. This is the objective function that
        we will want to minimize with respect to the weights, w.

        beta: Weight vector in Mx1 numpy ndarray6yy
        X: Design matrix in NxM numpy ndarray
        y: True output data in Nx1 numpy ndarray
        lamb: Scalar lambda value (sorry 'lambda' is a Python keyword)

        Returns: scalar value for the negative log likelihood
    """
    ######################
    #   YOUR CODE HERE   #
    ######################
    p = logistic(X, beta)
    l2 = np.sum(beta ** 2)
    NLL = -np.sum(y*np.log(p) + (1-y)*np.log(1-p)) + ((lamb/2) * l2)
    return NLL

###################################################
#   FEEL FREE TO WRITE ANY HELPER FUNCTIONS HERE  #
###################################################


def gradient_descent(X, y, lamb, alpha, beta0, num_iter):
    """ Implement gradient descent on the objective function using the
        parameters specified below. Return a list of weight vectors for
        each iteration starting with the initial w0.

        X: Design matrix in NxM numpy ndarray
        y: True output data in Nx1 numpy ndarray
        lamb: Scalar lambda value (sorry 'lambda' is a Python keyword)
        alpha: Scalar learning rate
        beta0: Initial weight vector in Mx1 numpy ndarray

        Returns: List of (num_iter+1) weight vectors, starting with w0
        and then the weight vectors after each of the num_iter iterations..
        Each element in the list should be an Mx1 numpy ndarray.
    """
    ######################
    #   YOUR CODE HERE   #
    ######################
    weights = []
    weights.append(beta0)
    for i in range(num_iter):
      y_pred = logistic(X,beta0)
      error = y_pred - y
      grad = np.dot(X.T, error) + (lamb) * beta0
      beta0 = beta0 - alpha * grad
      weights.append(beta0)
    return weights


if __name__ == "__main__":
    objective_test()
    gradient_descent_test()