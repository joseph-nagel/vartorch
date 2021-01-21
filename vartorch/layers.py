'''
Variational layers.

Summary
-------
The class 'VariationalLinear' establishes a linear layer suitable to variational inference.
One can specify independent Gaussian priors for the weights and biases.
A Gaussian mean field approximation is used as the variational posterior.
Its means and standard deviations (or a transform of them) are the learnable parameters.
When in sampling mode, the weights are randomly drawn from the posterior
Otherwise, the weights are set to their posterior means.
The KL divergence is computed during the forward pass when sampling is turned on.

The extensions 'VariationalLinearWithUncertainLogits' and 'VariationalLinearWithLearnableTemperature'
allow for a finer modeling and control of the encountered aleatoric uncertainties.
The former represents the logits as Gaussian variables with learnable means and standard deviations.
The latter performs an input-dependent temperature scaling on the logits.

'''

import torch
import torch.nn as nn
from .divergences import kl_div_pytorch
from .reparametrization import Reparametrize, reparametrize

class VariationalLinear(nn.Module):
    '''
    Variational linear layer.

    Summary
    -------
    This realizes a linear layer for variational inference.
    Both prior and variational posterior are Gaussians with diagonal covariances.
    Such a model captures parametric uncertainties.

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

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1,
                 bias_std=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.reset_parameters()
        self.sampling = True

    def reset_parameters(self):
        '''Reset the parameters randomly.'''
        self.w_mu = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.w_rho = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.b_mu = nn.Parameter(torch.randn(self.out_features))
        self.b_rho = nn.Parameter(torch.randn(self.out_features))

    @property
    def sampling(self):
        '''Get sampling mode.'''
        return self._sampling

    @sampling.setter
    def sampling(self, mode):
        '''Set sampling mode.'''
        self._sampling = mode

    def forward(self, X):
        if self.sampling: # sample from q
            w = reparametrize(self.w_mu, self.w_rho)
            b = reparametrize(self.b_mu, self.b_rho)
        else: # return mean of q
            w = self.w_mu
            b = self.b_mu
        y = nn.functional.linear(X, w, b)
        if self.sampling: # compute KL divergence
            self.kl_acc = kl_div_pytorch(self.w_mu, self.w_rho, self.weight_std) \
                        + kl_div_pytorch(self.b_mu, self.b_rho, self.bias_std)
        else: # do not compute KL divergence
            self.kl_acc = torch.tensor(0.0, device=y.device)
        return y

class VariationalLinearWithUncertainLogits(nn.Module):
    '''
    Variational linear layer with uncertain logits.

    Summary
    -------
    A variational linear layer with uncertain logits is implemented.
    Here, logits follow Gaussians with learnable means and standard deviations.
    This is supposed to represent aleatoric uncertainties.

    Parameters
    ----------
    See documentation of 'VariationalLinear'.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1,
                 bias_std=1):
        super().__init__()
        self.logits_mu = VariationalLinear(in_features=in_features,
                                           out_features=out_features,
                                           weight_std=weight_std,
                                           bias_std=bias_std)
        self.logits_rho = VariationalLinear(in_features=in_features,
                                            out_features=out_features,
                                            weight_std=weight_std,
                                            bias_std=bias_std)
        self.reparametrize = Reparametrize()

    def epistemic(self):
        '''Set epistemic mode.'''
        self.logits_mu.sampling = True
        self.logits_rho.sampling = True
        self.reparametrize.sampling = False

    def aleatoric(self):
        '''Set aleatoric mode.'''
        self.logits_mu.sampling = False
        self.logits_rho.sampling = False
        self.reparametrize.sampling = True

    def forward(self, X):
        mu = self.logits_mu(X)
        rho = self.logits_rho(X)
        y = self.reparametrize(mu, rho)
        return y

class VariationalLinearWithLearnableTemperature(nn.Module):
    '''
    Variational linear layer with learnable temperature.

    Summary
    -------
    This provides a variational linear layer with learnable temperature scaling.
    The temperature is modeled as an exponentiated input-dependent output.
    It might allow the model to adapt more flexibly to aleatoric uncertainties.

    Parameters
    ----------
    See documentation of 'VariationalLinear'.

    '''

    def __init__(self,
                 in_features,
                 out_features,
                 weight_std=1,
                 bias_std=1):
        super().__init__()
        self.logits = VariationalLinear(in_features=in_features,
                                        out_features=out_features,
                                        weight_std=weight_std,
                                        bias_std=bias_std)
        self.logtemp = VariationalLinear(in_features=in_features,
                                         out_features=1,
                                         weight_std=weight_std,
                                         bias_std=bias_std)

    def forward(self, X):
        logits = self.logits(X)
        logtemp = self.logtemp(X)
        temp = 1 + torch.exp(logtemp)
        y = logits / temp
        return y

