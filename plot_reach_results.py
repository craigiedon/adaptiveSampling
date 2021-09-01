import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open("results/reach_results_1000_fixed.json", "r") as f:
    results = json.load(f)

# plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots()
xs = np.arange(10, 110, step=10)

for k, v in results.items():
    # xs = range(len(v))
    c = next(ax._get_lines.prop_cycler)['color']
    ax.plot(xs, v, linestyle='-', color=c, label=k, linewidth=1, zorder=1)
    ax.scatter(xs, v, color='white', s=100, zorder=2)
    ax.scatter(xs, v, color=c, s=20, zorder=3)

vals = np.array(list(results.values()))
min_y = np.min(vals)
max_y = np.max(vals)

ax.spines['bottom'].set_bounds(min(xs), max(xs))
ax.spines['left'].set_bounds(min_y, max_y)

x_ticks = list(range(min(xs), max(xs)+1, 5))
# ax.xaxis.set_ticks(x_ticks)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.01))
ax.tick_params(direction='in')

ax.set_xlabel("Experiments")
ax.set_ylabel("RMSE")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.legend()
plt.show()
