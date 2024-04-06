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
data : Data tools.
analysis : Analysis tools.
divergence : Kullback-Leibler divergence.
layers : Variational layers.
reparam : Reparametrization issues.
variational : Model variationalization.
vis : Visualization tools.

'''

__COPYRIGHT__ = 'Copyright 2020-2024 Joseph Benjamin Nagel'


from . import (
    data,
    analysis,
    divergence,
    layers,
    reparam,
    variational,
    vis
)


from .data import (
    make_half_moons,
    MoonsDataModule,
    MNISTDataModule
)

from .analysis import anomaly_score, calibration_metrics

from .layers import VariationalLinear

from .variational import VariationalClassification

from .vis import plot_data_2d, plot_function_2d

