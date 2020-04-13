'''
Variational Bayesian inference with PyTorch.

Summary
-------
This package enables stochastic variational inference with PyTorch.
For the moment, prior distributions and variational posteriors
are multivariate Gaussians with diagonal covariance matrices.
More general extensions are envisaged for the future.

Modules
-------
confidence : Confidence well-calibration.
divergences : Kullback-Leibler divergences.
layers : Variational layers.
reparametrization : Reparametrization trick.
variationalize : Model variationalization.

'''

__COPYRIGHT__ = 'Copyright 2020 Joseph Benjamin Nagel'

# from . import confidence
# from . import divergences
# from . import layers
# from . import reparametrization
# from . import variationalize

from .confidence import accuracy_vs_confidence
from .layers import VariationalLinear
from .reparametrization import Reparametrize, reparametrize
from .variationalize import VariationalClassifier

