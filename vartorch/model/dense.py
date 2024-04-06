'''Dense variational classifier.'''

import torch.nn as nn

from ..layers import make_dense
from .base import VarClassifier


class DenseBlock(nn.Sequential):
    '''Multiple (serial) dense layers.'''

    def __init__(self,
                 num_features,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation=None,
                 drop_rate=None,
                 variational=False,
                 var_opts={}):

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_features) >= 2:
            num_layers = len(num_features) - 1
        else:
            raise ValueError('Number of features needs at least two entries')

        # assemble layers
        layers = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = make_dense(
                in_features,
                out_features,
                batchnorm=batchnorm,
                activation=activation if is_not_last else last_activation,
                drop_rate=drop_rate,
                variational=variational,
                var_opts=var_opts
            )

            layers.append(dense)

        # initialize module
        super().__init__(*layers)


class DenseVarClassifier(VarClassifier):
    '''
    Dense variational classifier.

    Parameters
    ----------
    num_features : list
        Feature numbers for dense layers.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : None or str
        Nonlinearity type.
    last_activation : None or str
        Final activation.
    drop_rate : float
        Dropout probability.
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

    def __init__(self,
                 num_features,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation=None,
                 drop_rate=None,
                 weight_std=1.0,
                 bias_std=1.0,
                 param_mode='log',
                 num_samples=1,
                 likelihood_type='Categorical',
                 lr=1e-04):

        # create variational model
        var_opts = {
            'weight_std': weight_std,
            'bias_std': bias_std,
            'param_mode': param_mode
        }

        model = DenseBlock(
            num_features=num_features,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation,
            drop_rate=drop_rate,
            variational=True,
            var_opts=var_opts
        )

        # initialize parent class
        super().__init__(
            model=model,
            num_samples=num_samples,
            likelihood_type=likelihood_type,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

