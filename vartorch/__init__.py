'''
Variational inference for Bayesian neural nets with PyTorch.

Summary
-------
This package enables stochastic variational inference with PyTorch.
It allows for training and employing Bayesian neural networks.
At the moment, the scope is limited to classification problems only.
Prior distributions and variational posteriors are represented as
multivariate Gaussians with diagonal covariance matrices.
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

