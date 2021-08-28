import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import GPy


def f_2d(x):
    first_dim = np.sum([i * np.cos((i + 1.0) * x[0] + i) for i in range(1, 6)])
    second_dim = np.sum([i * np.cos((i + 1.0) * x[1] + i) for i in range(1, 6)])
    return first_dim * second_dim


# The "Full" one
ns = 50
domain = (1.0, 3.0)
xs = np.linspace(domain[0], domain[1], num=ns)
X_2d = np.array(list(itertools.product(xs, xs)))
ys = np.array([f_2d(x) for x in X_2d])
fig = plt.figure()
ax_0 = fig.add_subplot(121)
c = ax_0.pcolormesh(X_2d.reshape((ns, ns, 2))[:, :, 0], X_2d.reshape((ns, ns, 2))[:, :, 1], ys.reshape((ns, ns)),
                    vmin=-10, vmax=10, shading='auto')
fig.colorbar(c, ax=ax_0)

# The "Reduced" values...
# ns = 10
# xs = np.linspace(domain[0], domain[1], num=ns)
# X_2d = np.array(list(itertools.product(xs, xs)))
# ys = np.array([f_2d(x) for x in X_2d])
# ax_1 = fig.add_subplot(132)
# c = ax_1.pcolormesh(X_2d.reshape((ns, ns, 2))[:, :, 0], X_2d.reshape((ns, ns, 2))[:, :, 1], ys.reshape((ns, ns)), vmin=-10, vmax=10, shading='auto')
# fig.colorbar(c, ax=ax_1)
# plt.tight_layout()
# plt.show()

kernel = GPy.kern.RBF(2, ARD=True)
xs_reduced = np.linspace(domain[0], domain[1], num=10)
X_train = np.array(list(itertools.product(xs_reduced, xs_reduced)))
y_train = np.array([f_2d(x) for x in X_train]).reshape(-1, 1)
gp = GPy.models.GPRegression(X_train, y_train, kernel)
gp.Gaussian_noise.variance = 0.0
gp.Gaussian_noise.variance.fix()
gp.optimize_restarts()
y_pred, y_vars = gp.predict_noiseless(X_2d)

ax_1 = fig.add_subplot(122)
c = ax_1.pcolormesh(X_2d.reshape((ns, ns, 2))[:, :, 0], X_2d.reshape((ns, ns, 2))[:, :, 1], y_pred.reshape((ns, ns)), vmin=-10, vmax=10, shading='auto')
fig.colorbar(c, ax=ax_1)

print(X_train.shape)
print(y_pred.shape)
plt.show()
