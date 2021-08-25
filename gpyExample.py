import GPy
import GPy.kern
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# GPy.plotting.change_plotting_library('plotly')

X = np.linspace(-4, 1, 5).reshape(-1, 1)
Y = 3.0 * (1.0 - X) ** 2 * np.exp(-X**2 - 1) - 10 * (X / 5.0 - X**3) * np.exp(-X**2)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X, Y, kernel)
display(m)

fig = m.plot()
plt.show()

m.optimize(messages=True)

fig = m.plot()
plt.show()

m.optimize_restarts(num_restarts=10)

fig = m.plot()
plt.show()

y_pred, vars = m.predict(np.array([0.1, 0.2, 0.3]).reshape(3, 1))
print(y_pred)
print(vars)

# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
