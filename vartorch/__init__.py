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
divergence : Kullback-Leibler divergence.
layers : Variational layers.
reparam : Reparametrization issues.
variational : Model variationalization.
vis : Visualization tools.

'''

__COPYRIGHT__ = 'Copyright 2020-2023 Joseph Benjamin Nagel'


from . import analysis
from . import divergence
from . import layers
from . import reparam
from . import variational
from . import vis


from .analysis import anomaly_score, calibration_metrics

from .layers import VariationalLinear

from .variational import VariationalClassification

from .vis import plot_data_2d, plot_function_2d

