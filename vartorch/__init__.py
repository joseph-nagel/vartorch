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
analysis : Analysis tools.
divergences : Kullback-Leibler divergences.
layers : Variational layers.
reparametrization : Reparametrization tricks.
variationalize : Model variationalization.

'''

__COPYRIGHT__ = 'Copyright 2020 Joseph Benjamin Nagel'

# from . import analysis
# from . import divergences
# from . import layers
# from . import reparametrization
# from . import variationalize

from .analysis import anomaly_score, calibration_metrics
from .layers import VariationalLinear
from .variationalize import VariationalClassification

