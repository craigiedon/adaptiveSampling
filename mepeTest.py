from typing import List

import numpy as np
from IPython.core.display import display
from matplotlib import pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import GPy

np.random.seed(1)


def f(x):
    """The function to predict."""
    return 3.0 * (1.0 - x) ** 2 * np.exp(-x**2 - 1) - 10 * (x / 5.0 - x**3) * np.exp(-x**2)


# def plot_gp_1d(gp, all_points, obs_x):
#     # Plot the function, the prediction, and the 95% condifence interval based on the MSE
#     plt.figure()
#     plt.plot(all_points, f(all_points), 'r:', label=r'$f(x) = x\, \sin(x)$')
#     plt.plot(obs_x, f(obs_x), 'r.', markersize=10, label='Observations')
#
#     y_pred, sigma = gp.predict(all_points, return_std=True)
#     plt.plot(all_points, y_pred, 'b-', label='Prediction')
#
#     # plt.fill(np.concatenate([all_points, all_points[::-1]]),
#     #          np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
#     #          alpha=0.5, fc='b', ec='None', label='95% confidence interval')
#
#     plt.fill_between(all_points.reshape(-1), y_pred.reshape(-1) - 1.96 * sigma, y_pred.reshape(-1) + 1.96 * sigma,
#                      alpha=0.5, fc='b', ec='None', label='95% confidence interval')
#
#     plt.xlabel('$x$')
#     plt.ylabel('$f(x)$')
#     # plt.ylim(-4, 20)
#     plt.legend(loc='upper left')
#     plt.show()


def index_of_closest(ref_point, points: List) -> int:
    dists = [np.linalg.norm(p - ref_point) for p in points]
    return np.argmin(dists)


def cross_validation_error(i: int, X: np.ndarray, y_trus: np.ndarray, candidate_points, kernel) -> float:
    assert len(X) == len(y_trus)
    X_sub_i = X[np.arange(len(X)) != i]
    y_trus_sub_i = y_trus[np.arange(len(y_trus)) != i]
    gp_sub_i = GPy.models.GPRegression(X_sub_i, y_trus_sub_i, kernel)
    gp_sub_i.optimize()
    y_pred, y_vars = gp_sub_i.predict_noiseless(np.array([X[i]]))
    err_cv = (y_trus[i].reshape(-1) - y_pred.reshape(-1)) ** 2
    assert len(err_cv) == 1
    return err_cv[0]


def run():
    # Generate original points by a space filling algorithm
    domain = (-4, 1)
    initial_points = np.atleast_2d(np.linspace(domain[0], domain[1], num=3)).T
    X = initial_points
    y_trus = np.array([f(x) for x in X])

    # Mesh the input space for evaluations of the real function, the prediction, and its MSE
    candidate_points = np.atleast_2d(np.linspace(domain[0], domain[1], 100)).T

    # Instantiate a Gaussian Process model
    kernel = GPy.kern.RBF(input_dim=1)
    gp = GPy.models.GPRegression(X, y_trus, kernel)
    gp.optimize_restarts(num_restarts=10)
    display(gp)
    print(X)

    # While the stopping criterion is not me
    budget = 5
    for q in range(budget):
        # plot_gp_1d(gp, candidate_points, X)
        # gp.plot()
        # plt.show()
        y_preds, cp_vars = gp.predict_noiseless(candidate_points)

        err_trus = [y_trus[i] - y_preds[i] for i in range(len(X))]
        err_cvs = [cross_validation_error(i, X, y_trus, candidate_points, kernel) for i in range(len(X))]
        # Calculate the CV error at each observed point using Eq (17)
        cp_cv_errs = np.array([err_cvs[index_of_closest(cp, X)] for cp in candidate_points.reshape(-1)])

        if q == 0:
            balance_factor = 0.5
        else:
            balance_factor = 0.99 * np.minimum(1.0, 0.5 * err_trus[-1] / err_cvs[-1])

        # Form EPE criterion in (23)
        expected_prediction_errs = balance_factor * cp_cv_errs + (1.0 - balance_factor) * cp_vars.reshape(-1)

        # Obtain new point by solving (25)
        new_point = candidate_points[np.argmax(expected_prediction_errs)]
        assert new_point not in X

        # Update information
        X = np.append(X, [new_point], axis=0)
        y_trus = np.append(y_trus, [f(new_point)], axis=0)

        # Refit GP
        gp = GPy.models.GPRegression(X, y_trus, kernel)
        gp.optimize_restarts(verbose=False, parallel=True)
        # gp.optimize_restarts(num_restarts=100)
        # gp.optimize()

    gp = GPy.models.GPRegression(X, y_trus, kernel)
    gp
    gp.optimize(messages=False)
    gp.optimize_restarts(verbose=False, parallel=True)
    gp.plot()
    plt.show()

    # Testing stage: Take 1000 test points in the domain, compute their true error, compute their predictions, then do average RMSE
    test_X = np.atleast_2d(np.linspace(domain[0], domain[1], 1000)).T
    test_ys_tru = np.array([f(x) for x in test_X])
    test_ys_mepe_pred, _ = gp.predict_noiseless(test_X)

    mepe_rmse = np.sqrt(np.average((test_ys_tru - test_ys_mepe_pred) ** 2))

    # Sanity check: Test against random selection?
    rand_picks = np.random.choice(len(candidate_points), budget, replace=False)
    random_X = np.concatenate([initial_points,  candidate_points[rand_picks]])
    random_ys_tru = np.array([f(x) for x in random_X])
    rand_gp = GPy.models.GPRegression(random_X, random_ys_tru, kernel)
    rand_gp.optimize_restarts()
    test_ys_random_pred, _ = rand_gp.predict_noiseless(test_X)

    rand_gp.plot()
    plt.show()

    random_rmse = np.sqrt(np.average((test_ys_tru - test_ys_random_pred) ** 2))

    print("MEPE Error: ", mepe_rmse)
    print("Random Error: ", random_rmse)
    # Sanity check 2: Test against low-discrepancy model?


run()
