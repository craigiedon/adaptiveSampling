import itertools
import time
from typing import List, Callable, Tuple
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from IPython.core.display import display
from matplotlib import pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import GPy

np.random.seed(1)


def simple_1d_func(x):
    return 3.0 * (1.0 - x) ** 2 * np.exp(-x ** 2 - 1) - 10 * (x / 5.0 - x ** 3) * np.exp(-x ** 2)


def plot_gp_1d(ax, gp: GPy.models.GPRegression, all_points, tru_func=None):
    if tru_func is not None:
        ax.plot(all_points, tru_func(all_points), color='r', label=r'Ground Truth')

    ax.plot(gp.X, gp.Y, 'kx', markersize=5, label='Observations')
    y_pred, y_vars = gp.predict_noiseless(all_points)
    y_sig = np.sqrt(y_vars).reshape(-1)
    ax.plot(all_points, y_pred, 'b-', label='Prediction')

    ax.fill_between(all_points.reshape(-1), y_pred.reshape(-1) - 1.96 * y_sig, y_pred.reshape(-1) + 1.96 * y_sig,
                    alpha=0.2, fc='b', ec='None', label='95% confidence interval')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend(loc='upper left')


def index_of_closest(ref_point, points: List) -> int:
    dists = [np.linalg.norm(p - ref_point) for p in points]
    return np.argmin(dists)


def cross_validation_error(i: int, X: np.ndarray, y_trus: np.ndarray) -> float:
    assert len(X) == len(y_trus)
    X_sub_i = X[np.arange(len(X)) != i]
    y_trus_sub_i = y_trus[np.arange(len(y_trus)) != i]
    gp_sub_i = noiseless_gp(X_sub_i, y_trus_sub_i)
    # gp_sub_i.optimize_restarts(verbose=False, parallel=True)
    gp_sub_i.optimize()
    y_pred, y_vars = gp_sub_i.predict_noiseless(np.array([X[i]]))
    err_cv = (y_trus[i].reshape(-1) - y_pred.reshape(-1)) ** 2
    assert len(err_cv) == 1
    return err_cv[0]


def cross_validation_error_fast(gp: GPy.models.GPRegression) -> List[float]:
    R = gp.kern.K(gp.X) / gp.kern.variance
    R_inv = np.linalg.inv(R)
    one_vec = np.ones((len(gp.X), 1))

    inv_11 = np.linalg.inv(one_vec.T @ one_vec)
    H = one_vec @ inv_11 @ one_vec.T

    R_inv_y = R_inv @ gp.Y
    inv_1r1 = np.linalg.inv(one_vec.T @ R_inv @ one_vec)
    mu_est = inv_1r1 @ one_vec.T @ R_inv_y
    d = gp.Y - one_vec @ mu_est

    rhs_mat = d.reshape(-1) + one_vec.reshape(-1) + H * d / (1.0 - H.diagonal())
    cv_err_fast = (np.sum(R_inv * rhs_mat, axis=1) / R_inv.diagonal()) ** 2.0
    # print("CV Fast: ", cv_err_fast)

    # for i in range(len(gp.X)):
    #     rhs = d.reshape(-1) + one_vec.reshape(-1) + H[:, i] * d[i] / (1.0 - H[i][i])
    #     cv_err_i = R_inv[i, :] @ rhs . R_inv[i,i]
    #     print(cv_err_i / R_inv[i, i])

    return cv_err_fast


def noiseless_gp(xs, ys) -> GPy.models.GPRegression:
    kernel = GPy.kern.Matern52(input_dim=xs.shape[-1])#, useGPU=True)
    gp = GPy.models.GPRegression(xs, ys, kernel)
    gp.Gaussian_noise.variance = 0.0
    gp.Gaussian_noise.variance.fix()
    return gp


def mepe_sampling(budget: int, initial_points: np.ndarray, candidate_points: np.ndarray, sim_func: Callable, plot_intermediates=False) -> GPy.models.GPRegression:
    assert initial_points.ndim >= 2
    assert candidate_points.ndim == initial_points.ndim
    assert budget > 0

    X = initial_points
    y_trus = np.array([sim_func(x) for x in X]).reshape(-1, 1)

    # Instantiate a Gaussian Process model
    gp = noiseless_gp(X, y_trus)
    gp.optimize_restarts(verbose=False, parallel=True)

    # While the stopping criterion is not me
    for q in range(budget):
        print("\tIteration: ", q, end='\t')
        start_time = time.time()
        if plot_intermediates:
            gp.plot()
            plt.title(f"Choice {q}")
            plt.show()
        y_preds, cp_vars = gp.predict_noiseless(candidate_points)

        err_trus = [y_trus[i] - y_preds[i] for i in range(len(X))]
        # err_cvs = [cross_validation_error(i, X, y_trus) for i in range(len(X))]
        err_cvs = cross_validation_error_fast(gp)
        # Calculate the CV error at each observed point using Eq (17)
        cp_cv_errs = np.array([err_cvs[index_of_closest(cp, X)] for cp in candidate_points])

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
        y_trus = np.append(y_trus, np.array([sim_func(new_point)]).reshape(-1, 1), axis=0)

        # Refit GP
        gp = noiseless_gp(X, y_trus)
        gp.optimize_restarts(verbose=False, parallel=True)
        print("Time: ", time.time() - start_time)

    gp = noiseless_gp(X, y_trus)
    gp.optimize_restarts(verbose=False, parallel=True)

    return gp


def run():
    # Generate original points by a space filling algorithm
    domain = (-4, 1)
    initial_points = np.atleast_2d(np.linspace(domain[0], domain[1], num=3)).T
    candidate_points = np.atleast_2d(np.linspace(domain[0], domain[1], 100)).T
    budget = 7
    gp, X = mepe_sampling(budget, initial_points, candidate_points, simple_1d_func, False)

    fig, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2)

    # Testing stage: Take 1000 test points in the domain, compute their true error, compute their predictions, then do average RMSE
    test_X = np.atleast_2d(np.linspace(domain[0], domain[1], 1000)).T
    test_ys_tru = np.array([simple_1d_func(x) for x in test_X])
    test_ys_mepe_pred, test_ys_mepe_vars = gp.predict_noiseless(test_X)

    mepe_rmse = np.sqrt(np.average((test_ys_tru - test_ys_mepe_pred) ** 2))

    test_ys_mepe_std = np.sqrt(test_ys_mepe_vars)

    plot_gp_1d(ax_0, gp, test_X, simple_1d_func)
    ax_0.set_title("MEPE Final")

    # Sanity check: Test against random selection?
    rand_picks = np.random.choice(len(candidate_points), budget, replace=False)
    random_X = np.concatenate([initial_points, candidate_points[rand_picks]])
    random_ys_tru = np.array([simple_1d_func(x) for x in random_X])
    rand_gp = noiseless_gp(random_X, random_ys_tru)
    rand_gp.optimize_restarts(verbose=False, parallel=True)
    test_ys_random_pred, test_ys_random_var = rand_gp.predict_noiseless(test_X)
    test_ys_random_std = np.sqrt(test_ys_random_var)

    # rand_gp.plot()
    plot_gp_1d(ax_1, rand_gp, test_X, simple_1d_func)
    ax_1.set_title("Random")
    plt.show()

    random_rmse = np.sqrt(np.average((test_ys_tru - test_ys_random_pred) ** 2))

    print("MEPE Error: ", mepe_rmse)
    print("Random Error: ", random_rmse)
    # Sanity check 2: Test against low-discrepancy model?


if __name__ == "__main__":
    run()
