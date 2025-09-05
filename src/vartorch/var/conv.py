'''Conv. variational classifier.'''

from collections.abc import Sequence

import torch.nn as nn

from ..layers import IntOrInts, ActivType
from ..model import ConvDown, DenseBlock
from .base import VarClassifier


class ConvVarClassifier(VarClassifier):
    '''
    Conv. variational classifier.

    Parameters
    ----------
    num_channels : list
        Channel numbers for conv. layers.
    num_features : list
        Feature numbers for dense layers.
    kernel_size : int or (int, int)
        Conv. kernel size.
    pooling : int, (int, int) or None
        Pooling parameter.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str or None
        Nonlinearity type.
    last_activation : str or None
        Nonlinearity of the final layer.
    drop_rate : float or None
        Dropout probability for dense layers.
    pool_last : bool
        Controls the last pooling operation (also first upscaling).
    double_conv : bool
        Determines whether double conv. blocks are used.
    weight_std : float
        Prior std. of the weights.
    bias_std : float
        Prior std. of the biases.
    param_mode : {'log', 'rho'}
        Determines how the non-negative standard deviation
        is represented in terms of a real-valued parameter.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'Categorical'}
        Likelihood function type.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        num_features: Sequence[int],
        kernel_size: IntOrInts = 3,
        pooling: IntOrInts | None = 2,
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = None,
        drop_rate: float | None = None,
        pool_last: bool = True,
        double_conv: bool = True,
        weight_std: float = 1.0,
        bias_std: float = 1.0,
        param_mode: str = 'log',
        num_samples: int = 1,
        likelihood_type: str = 'Categorical',
        lr: float = 1e-04
    ) -> None:

        # check feature numbers
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        # get number of classes
        if likelihood_type == 'Bernoulli':
            if num_features[-1] == 1:
                num_classes = 2
            else:
                ValueError('Bernoulli likelihood requires a single output')

        elif likelihood_type == 'Categorical':
            if num_features[-1] > 1:
                num_classes = num_features[-1]
            else:
                ValueError('Categorical likelihood requires multiple outputs')

        else:
            raise ValueError(f'Unknown likelihood type: {likelihood_type}')

        # create conv layers
        conv_layers = ConvDown(
            num_channels=num_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1,
            pooling=pooling,
            batchnorm=batchnorm,
            activation=activation,
            last_activation='same',
            normalize_last=True,
            pool_last=pool_last,
            double_conv=double_conv,
            inout_first=True
        )

        # create dense variational layers
        var_opts = {
            'weight_std': weight_std,
            'bias_std': bias_std,
            'param_mode': param_mode
        }

        dense_layers = DenseBlock(
            num_features=num_features,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation,
            normalize_last=False,
            drop_rate=drop_rate,
            variational=True,
            var_opts=var_opts
        )

        # assemble model
        model = nn.Sequential(
            conv_layers,
            nn.Flatten(start_dim=1),
            dense_layers
        )

        # initialize parent class
        super().__init__(
            model=model,
            num_samples=num_samples,
            likelihood_type=likelihood_type,
            num_classes=num_classes,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)
