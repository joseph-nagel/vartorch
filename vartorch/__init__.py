'''
Variational inference for Bayesian neural nets with PyTorch.

Summary
-------
This package enables stochastic variational inference with PyTorch.
It allows for training and employing Bayesian neural networks.
At the moment, the scope is limited to classification problems only.
Prior distributions and variational posteriors are represented as
multivariate Gaussians with diagonal covariance matrices.

Modules
-------
data : Data tools.
kldiv : Kullback-Leibler divergence.
layers : Variational layers.
model : Variational mdoels.
metrics : Analysis tools.
reparam : Reparametrization issues.
vis : Visualization tools.

'''

from . import (
    data,
    kldiv,
    layers,
    model,
    metrics,
    reparam,
    vis
)


from .data import (
    make_half_moons,
    MoonsDataModule,
    MNISTDataModule
)

from .kldiv import (
    kl_div_dist,
    kl_div_analytical,
    kl_div_mc
)

from .layers import (
    VarLayer,
    VarLinear,
    VarLinearWithUncertainLogits,
    VarLinearWithLearnableTemperature,
    Reparametrize,
    make_activation,
    make_block,
    make_dropout,
    make_dense,
    make_conv
)

from .model import (
    VarClassifier,
    DenseBlock,
    DenseVarClassifier
)

from .metrics import anomaly_score, calibration_metrics

from .reparam import (
    reparametrize,
    sigma_from_log,
    sigma_from_rho
)

from .vis import plot_data_2d, plot_function_2d

