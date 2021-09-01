import json
import random
from collections import defaultdict

import numpy as np
import probRobScene
from probRobScene.core.experiment_design import try_unif_design
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

from mepeTest import noiseless_gp, mepe_sampling
from stl import F, LEQ0, agm_rob


def closest_traj(cube_xy: np.ndarray, candidate_points: np.ndarray, y_tru: np.ndarray) -> float:
    closest_i = np.argmin([np.linalg.norm(cube_xy - cp) for cp in candidate_points])
    return y_tru[closest_i]


def run():
    epsilon = 0.05
    dist_func = lambda state: np.minimum(1.0, np.linalg.norm(np.array(state[0]) - np.array(state[1]))) - epsilon
    spec = F(LEQ0(dist_func), 0, 99)

    scenario = probRobScene.scenario_from_file("scenarios/cubeOnTable.prs")
    with open("simulation_traces/cube_reach_1000.json", "r") as f:
        sim_traces = json.load(f)

    candidate_points = np.array([s["input_state"] for s in sim_traces])[:, :2]
    initial_points = candidate_points[np.arange(0, len(candidate_points), step=int(len(candidate_points) / 10))]
    y_tru = np.array([agm_rob(spec, s["trace"], 0) for s in sim_traces]).reshape(-1, 1)

    # budgets = [10, 25, 50]
    # budgets = np.arange(10, 105, step=5)
    budgets = np.arange(10, 110, step=10)

    rmses_data = defaultdict(list)

    prev_mepe_candidates = initial_points
    prev_budget = 0
    for budget in budgets:
        print("Current Budget: ", budget)
        ### Training Part

        gp_mepe = mepe_sampling(budget=budget - prev_budget, initial_points=prev_mepe_candidates,
                                candidate_points=candidate_points, sim_func=lambda x: closest_traj(x, candidate_points, y_tru), plot_intermediates=False)
        prev_mepe_candidates = gp_mepe.X
        prev_budget = budget

        ud_scenes = try_unif_design(scenario, budget)
        ud_budget_points = np.array([s.objects[2].position.coordinates for s in ud_scenes])[:, :2]
        ud_points = np.concatenate([initial_points, ud_budget_points])
        ud_ys_tru = np.array([closest_traj(x, candidate_points, y_tru) for x in ud_points])
        gp_ud = noiseless_gp(ud_points, ud_ys_tru)
        gp_ud.optimize_restarts(verbose=False, parallel=True)

        # Average Random Model over 10 iterations for fairness
        random_gps = []
        for i in range(10):
            random_indices = np.random.choice(range(0, len(candidate_points)), size=budget, replace=False)
            random_points = np.concatenate([initial_points, candidate_points[random_indices]])
            random_ys_tru = np.array([closest_traj(x, candidate_points, y_tru) for x in random_points])
            gp_random = noiseless_gp(random_points, random_ys_tru)
            gp_random.optimize_restarts(verbose=False, parallel=True)
            random_gps.append(gp_random)

        gp_random.save_model(f"results/random_gp_{budget}")
        gp_ud.save_model(f"results/ud_gp_{budget}")
        gp_mepe.save_model(f"results/mepe_gp_{budget}")

        ### Testing Part
        test_ys_mepe_pred, test_ys_mepe_var = gp_mepe.predict_noiseless(candidate_points)
        test_ys_mepe_std = np.sqrt(test_ys_mepe_var)
        mepe_rmse = np.sqrt(np.average((y_tru - test_ys_mepe_pred) ** 2))

        random_rmses = []
        for gpr in random_gps:
            test_ys_random_pred, test_ys_random_var = gpr.predict_noiseless(candidate_points)
            random_rmses.append(np.sqrt(np.average((y_tru - test_ys_random_pred) ** 2)))

        # print("Rands: ", random_rmses)
        random_rmse = np.average(random_rmses)

        test_ys_ud_pred, test_ys_ud_var = gp_ud.predict_noiseless(candidate_points)
        test_ys_ud_std = np.sqrt(test_ys_ud_var)
        ud_rmse = np.sqrt(np.average((y_tru - test_ys_ud_pred) ** 2))

        rmses_data["mepe"].append(mepe_rmse)
        rmses_data["random"].append(random_rmse)
        rmses_data["ud"].append(ud_rmse)

        print(f"Mepe RMSE {mepe_rmse}")
        print(f"Random RMSE {random_rmse}")
        print(f"Uniform Design RMSE {ud_rmse}")

    # Saving the data somewhere
    print(rmses_data)
    # with open("results/reach_results_1000_granular.json", 'w') as f:
    #     json.dump(rmses_data, f)


    # gp_full = noiseless_gp(candidate_points, y_tru)
    # gp_full.optimize_restarts(verbose=False, parallel=True)
    # test_ys_full_pred, test_ys_full_var = gp_full.predict_noiseless(candidate_points)
    # test_ys_full_std = np.sqrt(test_ys_full_var)
    # full_rmse = np.sqrt(np.average((y_tru - test_ys_full_pred) ** 2))
    # # SANITY CHECK: Should be 0!
    # print(f"Full GP RMSE:  {full_rmse}")

    ### Plotting part
    # fig = plt.figure()
    #
    # xi = np.linspace(np.min(candidate_points[:, 0]), np.max(candidate_points[:, 0]), 50)
    # yi = np.linspace(np.min(candidate_points[:, 1]), np.max(candidate_points[:, 1]), 50)
    # xi, yi = np.meshgrid(xi, yi)
    # val_interp = Rbf(candidate_points[:, 0], candidate_points[:, 1], y_tru.reshape(-1), function='linear')
    #
    # zi_tru = val_interp(xi, yi)
    #
    # ax_0 = fig.add_subplot(131)
    # c = ax_0.pcolormesh(xi, yi, zi_tru, vmin=-0.5, vmax=0.1)
    # fig.colorbar(c, ax=ax_0)
    #
    # ax_1 = fig.add_subplot(132)
    # zi_mepe_pred = gp_mepe.predict_noiseless(np.stack((xi, yi), axis=2).reshape(-1, 2))[0].reshape(50, 50)
    # c = ax_1.pcolormesh(xi, yi, zi_mepe_pred, vmin=-0.5, vmax=0.1)
    # fig.colorbar(c, ax=ax_1)
    #
    # ax_2 = fig.add_subplot(133)
    # zi_random_pred = gp_random.predict_noiseless(np.stack((xi, yi), axis=2).reshape(-1, 2))[0].reshape(50, 50)
    # c = ax_2.pcolormesh(xi, yi, zi_random_pred, vmin=-0.5, vmax=0.1)
    # fig.colorbar(c, ax=ax_2)
    #
    # plt.show()


if __name__ == "__main__":
    run()
