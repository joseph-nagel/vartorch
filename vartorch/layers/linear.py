'''
Variational linear layers.

Summary
-------
The child class 'VarLinear' establishes a linear layer suitable to variational inference.
One can specify independent Gaussian priors for the weights and biases.
A Gaussian mean field approximation is used as the variational posterior.
Its means and standard deviations (or a transform of them) are the learnable parameters.
When in sampling mode, the weights are randomly drawn from the posterior
Otherwise, the weights are set to their posterior means.
The KL divergence is computed during the forward pass when sampling is turned on.

The extensions 'VarLinearWithUncertainLogits' and 'VarLinearWithLearnableTemperature'
allow for a finer modeling and control of the encountered uncertainties.
The former represents the logits as Gaussians with learnable means and standard deviations.
The latter performs an input-dependent temperature scaling on the logits.

'''

import torch
import torch.nn as nn

from ..kldiv import kl_div_dist
from ..reparam import reparametrize
from .base import VarLayer
from .reparam import Reparametrize


class VarLinear(VarLayer):
    '''
    Variational linear layer.

    Summary
    -------
    This realizes a linear layer for variational inference.
    Both prior and variational posterior are Gaussians with diagonal covariances.
    Such a model captures epistemic parametric uncertainties.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    weight_std : float
        Prior std. of the weights.
    bias_std : float
        Prior std. of the biases.
    param_mode : {'log', 'rho'}
        Determines how the non-negative standard deviation
        is represented in terms of a real-valued parameter.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1.0,
                 bias_std=1.0,
                 param_mode='log'):

        super().__init__(param_mode)

        self.in_features = in_features
        self.out_features = out_features

        self.weight_std = weight_std
        self.bias_std = bias_std

        self.reset_parameters()

    def reset_parameters(self):
        '''Reset the parameters randomly.'''
        self.w_mu = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.w_sigma_param = nn.Parameter(torch.randn(self.out_features, self.in_features))

        self.b_mu = nn.Parameter(torch.randn(self.out_features))
        self.b_sigma_param = nn.Parameter(torch.randn(self.out_features))

    def forward(self, X):

        # sample from q
        if self.sampling:
            w_sigma = self.sigma(self.w_sigma_param)
            b_sigma = self.sigma(self.b_sigma_param)

            w = reparametrize(self.w_mu, w_sigma)
            b = reparametrize(self.b_mu, b_sigma)

        # return mean of q
        else:
            w = self.w_mu
            b = self.b_mu

        y = nn.functional.linear(X, w, b)

        # compute KL divergence
        if self.sampling:
            self.kl_acc = kl_div_dist(self.w_mu, w_sigma, self.weight_std) \
                        + kl_div_dist(self.b_mu, b_sigma, self.bias_std)
        # do not compute KL divergence
        else:
            self.kl_acc = torch.tensor(0.0, device=y.device)

        return y


class VarLinearWithUncertainLogits(VarLayer):
    '''
    Variational linear layer with uncertain logits.

    Summary
    -------
    A variational linear layer with uncertain logits is implemented.
    Here, logits follow Gaussians with learnable means and standard deviations.
    This is supposed to represent aleatoric uncertainties.

    Parameters
    ----------
    See documentation of 'VarLinear'.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1.0,
                 bias_std=1.0,
                 param_mode='log'):

        super().__init__(param_mode)

        self.logits_mu = VarLinear(
            in_features=in_features,
            out_features=out_features,
            weight_std=weight_std,
            bias_std=bias_std,
            param_mode=param_mode

        )

        self.logits_sigma_param = VarLinear(
            in_features=in_features,
            out_features=out_features,
            weight_std=weight_std,
            bias_std=bias_std,
            param_mode=param_mode
        )

        self.reparametrize = Reparametrize(param_mode=param_mode)

    def epistemic(self):
        '''Set epistemic mode.'''
        self.logits_mu.sampling = True
        self.logits_sigma_param.sampling = True
        self.reparametrize.sampling = False

    def aleatoric(self):
        '''Set aleatoric mode.'''
        self.logits_mu.sampling = False
        self.logits_sigma_param.sampling = False
        self.reparametrize.sampling = True

    def forward(self, X):
        mu = self.logits_mu(X)
        sigma_param = self.logits_sigma_param(X)
        sigma = self.sigma(sigma_param)
        y = self.reparametrize(mu, sigma)
        return y


class VarLinearWithLearnableTemperature(VarLayer):
    '''
    Variational linear layer with learnable temperature.

    Summary
    -------
    This provides a variational linear layer with learnable temperature scaling.
    The temperature is modeled as an exponentiated input-dependent output.
    It might allow the model to adapt more flexibly to aleatoric uncertainties.

    Parameters
    ----------
    See documentation of 'VarLinear'.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1.0,
                 bias_std=1.0,
                 param_mode='log'):

        super().__init__(param_mode)

        self.logits = VarLinear(
            in_features=in_features,
            out_features=out_features,
            weight_std=weight_std,
            bias_std=bias_std,
            param_mode=param_mode
        )

        self.logtemp = VarLinear(
            in_features=in_features,
            out_features=1,
            weight_std=weight_std,
            bias_std=bias_std,
            param_mode=param_mode
        )

    def forward(self, X):
        logits = self.logits(X)
        logtemp = self.logtemp(X)
        temp = torch.exp(logtemp)
        # temp = 1 + torch.exp(logtemp)
        y = logits / temp
        return y

