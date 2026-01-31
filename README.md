# Variational inference for Bayesian neural nets

This project implements variational inference for Bayesian neural networks with PyTorch.
While the computational expense is expected to increase in comparison to classical model training,
the approach enables a means of uncertainty quantification in deep learning.
Only classification problems can be addressed at this point.
Another limitation is that the variational distribution, which acts as a parametric posterior approximation,
is restricted to a multivariate Gaussian with a diagonal covariance matrix.

<p>
  <img src="assets/bnn.png" alt="Bayesian neural network with probabilistic weights" title="Bayesian neural network" height="300" style="padding-right: 1em;">
  <img src="assets/uncertainty.svg" alt="Uncertainty of the posterior predictions" title="Posterior prediction uncertainty" height="300">
</p>


## Notebooks

- [Half-moons example](notebooks/moons.ipynb)
- [MNIST example](notebooks/mnist.ipynb)


## Installation

```bash
pip install -e .
```


## Training

```bash
python scripts/main.py fit --config config/moons.yaml
```

```bash
python scripts/main.py fit --config config/mnist.yaml
```
