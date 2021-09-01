import json

import GPy
import probRobScene
from GPy import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

from stl import F, LEQ0, agm_rob

epsilon = 0.05
dist_func = lambda state: np.minimum(1.0, np.linalg.norm(np.array(state[0]) - np.array(state[1]))) - epsilon
spec = F(LEQ0(dist_func), 0, 99)
with open("simulation_traces/cube_reach_1000.json", "r") as f:
    sim_traces = json.load(f)

candidate_points = np.array([s["input_state"] for s in sim_traces])[:, :2]
initial_points = candidate_points[np.arange(0, len(candidate_points), step=int(len(candidate_points) / 10))]
y_tru = np.array([agm_rob(spec, s["trace"], 0) for s in sim_traces]).reshape(-1, 1)

for budget in [50]: #np.arange(10, 110, step=10):
    gp_mepe = Model.load_model(f"results/mepe_gp_{budget}.zip")
    gp_random = Model.load_model(f"results/random_gp_{budget}.zip")
    gp_ud = Model.load_model(f"results/ud_gp_{budget}.zip")

    ### Plotting part
    fig, (ax_0, ax_1, ax_2, ax_3) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(4, 6), squeeze=True, constrained_layout=True, subplot_kw={"aspect": "equal"})

    len(candidate_points)

    ns = 33
    xi = np.linspace(np.min(candidate_points[:, 0]), np.max(candidate_points[:, 0]), ns)
    yi = np.linspace(np.min(candidate_points[:, 1]), np.max(candidate_points[:, 1]), ns)
    xi, yi = np.meshgrid(xi, yi)
    val_interp = Rbf(candidate_points[:, 0], candidate_points[:, 1], y_tru.reshape(-1), function='linear')

    zi_tru = val_interp(xi, yi)

    c = ax_0.pcolormesh(xi, yi, zi_tru, vmin=-0.5, vmax=0.1)
    ax_0.set_title("Ground Truth")

    zi_mepe_pred = gp_mepe.predict_noiseless(np.stack((xi, yi), axis=2).reshape(-1, 2))[0].reshape(ns, ns)
    c = ax_1.pcolormesh(xi, yi, zi_mepe_pred, vmin=-0.5, vmax=0.1)
    ax_1.set_title(f"MEPE")

    zi_random_pred = gp_random.predict_noiseless(np.stack((xi, yi), axis=2).reshape(-1, 2))[0].reshape(ns, ns)
    c = ax_2.pcolormesh(xi, yi, zi_random_pred, vmin=-0.5, vmax=0.1)
    ax_2.set_title(f"Random")

    zi_ud_pred = gp_ud.predict_noiseless(np.stack((xi, yi), axis=2).reshape(-1, 2))[0].reshape(ns, ns)
    c = ax_3.pcolormesh(xi, yi, zi_ud_pred, vmin=-0.5, vmax=0.1)
    ax_3.set_title(f"Uniform")

    c_bar = fig.colorbar(c, ax=[ax_0, ax_1, ax_2, ax_3])
    c_bar.set_label("Robustness Score")

    plt.show()
