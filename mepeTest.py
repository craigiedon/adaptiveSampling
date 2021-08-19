import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)


def plot_gp(gp, all_points, obs_x):
    # Plot the function, the prediction, and the 95% condifence interval based on the MSE
    plt.figure()
    plt.plot(all_points, f(all_points), 'r:', label=r'$f(x) = x\, \sin(x)$')
    plt.plot(obs_x, f(obs_x), 'r.', markersize=10, label='Observations')

    y_pred, sigma = gp.predict(all_points, return_std=True)
    plt.plot(all_points, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([all_points, all_points[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma,
                             (y_pred + 1.96 * sigma)[::-1]]),
             alpha=0.5, fc='b', ec='None', label='95% confidence interval')

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


def run():
    # Generate original points by a space filling algorithm
    X = np.atleast_2d(np.linspace(0, 10, num=3)).T

    # Evaluate the responses to each of the initial points
    y = f(X).ravel()

    # Mesh the input space for evaluations of the real function, the prediction, and its MSE
    candidate_points = np.atleast_2d(np.linspace(0, 10, 100)).T

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1.0, 100))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(candidate_points, return_std=True)
    plot_gp(gp, candidate_points, X)

    voronoi_cells = []

    # Calculate
    print(X)
    err_cvs = []
    for i in range(len(X)):
        # All items *except* the one at index i
        X_sub_i = X[np.arange(len(X)) != i]
        y_sub_i = y[np.arange(len(y)) != i]
        gp_sub_i = GaussianProcessRegressor(kernel, n_restarts_optimizer=9).fit(X_sub_i, y_sub_i)
        plot_gp(gp_sub_i, candidate_points, X_sub_i)
        i_pred, i_sigma = gp_sub_i.predict([X[i]], return_std=True)
        err_cv = np.power(y[i] - i_pred, 2)
        err_cvs.append(err_cv)
    print("CV Errors: ", err_cvs)

    # While the stopping criterion is not me
    chosen_points = []
    for i in range(10):
        for cp in candidate_points:
            # Find index of closest chosen points
            closest_i = index_of_closest()

            pass
            # print(cp)
        # Use Eq 24 to update balance factor alpha
        # if i == 0:
        #     balance_factor = 0.5
        # else:
        #     balance_factor = np.minimum(1.0, 0.5 * err_tru(chosen_points[-1]) / err_cv(chosen_points[-1]))
        #
        # err_cv = closest_voronoi_cell(vornoi_cells
        # Calculate the CV error at each observed point using Eq 17
        # Use local exp term (21), glob exp term (10) and balance factor to form (23)
        # Obtain new point by solving (25)
        # Call f to obtain response of new point
        # Update information
        # Refit GP


run()
