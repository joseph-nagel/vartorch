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
metrics : Analysis tools.
model : Model blocks.
reparam : Reparametrization issues.
var : Variational models.
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
    make_conv,
    SingleConv,
    DoubleConv
)

from .metrics import anomaly_score, calibration_metrics

from .model import DenseBlock

from .reparam import (
    reparametrize,
    sigma_from_log,
    sigma_from_rho,
    log_from_sigma,
    rho_from_sigma

)

from .var import (
    VarClassifier,
    ConvVarClassifier,
    DenseVarClassifier
)

from .vis import (
    plot_point_predictions,
    plot_post_predictions,
    plot_entropy_histograms,
    plot_data_2d,
    plot_function_2d,
    plot_data_and_preds_2d,
    point_prediction,
    post_mean,
    post_predictive,
    post_uncertainty
)

