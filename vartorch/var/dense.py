'''Dense variational classifier.'''

from ..model import DenseBlock
from .base import VarClassifier


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
        Nonlinearity of the final layer.
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
            normalize_last=False,
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

