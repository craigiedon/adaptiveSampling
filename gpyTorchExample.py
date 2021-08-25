import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(-4, 1, 5)
# True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) #+ torch.randn(train_x.size()) * math.sqrt(0.04)
train_y = 3.0 * (1.0 - train_x) ** 2 * torch.exp(-train_x**2 - 1) - 10 * (train_x / 5.0 - train_x**3) * torch.exp(-train_x**2)

# We will use the simplest form of GP Model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# Loss for GPs - marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 1000
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}   noise: {model.likelihood.noise.item():.3f}")
    optimizer.step()

model.eval()
likelihood.eval()
test_x = torch.linspace(-4, 1, 100)
test_y = 3.0 * (1.0 - test_x) ** 2 * torch.exp(-test_x**2 - 1) - 10 * (test_x / 5.0 - test_x**3) * torch.exp(-test_x**2)
observed_pred = likelihood(model(test_x))
f_preds = model(test_x)

f_mean = f_preds.mean
f_var = f_preds.variance
f_covar = f_preds.covariance_matrix
f_samples = f_preds.sample(sample_shape=torch.Size((1000,)))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    ax.plot(test_x.numpy(), test_y.numpy(), 'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()
