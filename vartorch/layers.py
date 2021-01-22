'''
Variational layers.

Summary
-------
A base class for variational layers is defined in 'VariationalLayer'.
This manages the main settings regarding the probabilistic weights.
In particular, it comes with a parametrization mode and a sampling switch.

The child class 'VariationalLinear' establishes a linear layer suitable to variational inference.
One can specify independent Gaussian priors for the weights and biases.
A Gaussian mean field approximation is used as the variational posterior.
Its means and standard deviations (or a transform of them) are the learnable parameters.
When in sampling mode, the weights are randomly drawn from the posterior
Otherwise, the weights are set to their posterior means.
The KL divergence is computed during the forward pass when sampling is turned on.

The extensions 'VariationalLinearWithUncertainLogits' and 'VariationalLinearWithLearnableTemperature'
allow for a finer modeling and control of the encountered uncertainties.
The former represents the logits as Gaussian variables with learnable means and standard deviations.
The latter performs an input-dependent temperature scaling on the logits.

'Reparametrize' implements the reparametrization trick for the MC estimation of the ELBO loss.
It can be used as component for building advanced layers.

'''

import torch
import torch.nn as nn
from .divergences import kl_div_pytorch
from .reparametrization import \
    reparametrize, sigma_from_log, sigma_from_rho

class VariationalLayer(nn.Module):
    '''
    Variational layer base class.

    Summary
    -------
    Variational layers can inherit from this parent class.
    It manages how standard deviations of the variational distribution are parametrized
    and allows for turning random weight sampling on and off for the predictions.

    Parameters
    ----------
    param_mode : {'log', 'rho'}
        Determines how the non-negative standard deviation
        is represented in terms of a real-valued parameter.

    '''

    def __init__(self, param_mode='log'):
        super().__init__()
        self.parametrization = param_mode
        self.sampling = True

    @property
    def parametrization(self):
        '''Get parametrization mode.'''
        return self._parametrization

    @parametrization.setter
    def parametrization(self, param_mode):
        '''Set parametrization mode.'''
        self._parametrization = param_mode
        if param_mode == 'log':
            self.sigma = sigma_from_log
        elif param_mode == 'rho':
            self.sigma = sigma_from_rho

    @property
    def sampling(self):
        '''Get sampling mode.'''
        return self._sampling

    @sampling.setter
    def sampling(self, sample_mode):
        '''Set sampling mode.'''
        self._sampling = sample_mode

class VariationalLinear(VariationalLayer):
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
                 weight_std=1.,
                 bias_std=1.,
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
        if self.sampling: # sample from q
            w_sigma = self.sigma(self.w_sigma_param)
            b_sigma = self.sigma(self.b_sigma_param)
            w = reparametrize(self.w_mu, w_sigma)
            b = reparametrize(self.b_mu, b_sigma)
        else: # return mean of q
            w = self.w_mu
            b = self.b_mu
        y = nn.functional.linear(X, w, b)
        if self.sampling: # compute KL divergence
            self.kl_acc = kl_div_pytorch(self.w_mu, w_sigma, self.weight_std) \
                        + kl_div_pytorch(self.b_mu, b_sigma, self.bias_std)
        else: # do not compute KL divergence
            self.kl_acc = torch.tensor(0.0, device=y.device)
        return y

class VariationalLinearWithUncertainLogits(VariationalLayer):
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
                 weight_std=1.,
                 bias_std=1.,
                 param_mode='log'):
        super().__init__(param_mode)
        self.logits_mu = VariationalLinear(in_features=in_features,
                                           out_features=out_features,
                                           weight_std=weight_std,
                                           bias_std=bias_std,
                                           param_mode=param_mode)
        self.logits_sigma_param = VariationalLinear(in_features=in_features,
                                                    out_features=out_features,
                                                    weight_std=weight_std,
                                                    bias_std=bias_std,
                                                    param_mode=param_mode)
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
        sigma = self.sigm(sigma_param)
        y = self.reparametrize(mu, sigma)
        return y

class VariationalLinearWithLearnableTemperature(VariationalLayer):
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
                 weight_std=1.,
                 bias_std=1.,
                 param_mode='log'):
        super().__init__(param_mode)
        self.logits = VariationalLinear(in_features=in_features,
                                        out_features=out_features,
                                        weight_std=weight_std,
                                        bias_std=bias_std,
                                        param_mode=param_mode)
        self.logtemp = VariationalLinear(in_features=in_features,
                                         out_features=1,
                                         weight_std=weight_std,
                                         bias_std=bias_std,
                                         param_mode=param_mode)

    def forward(self, X):
        logits = self.logits(X)
        logtemp = self.logtemp(X)
        temp = torch.exp(logtemp)
        # temp = 1 + torch.exp(logtemp)
        y = logits / temp
        return y

class Reparametrize(VariationalLayer):
    '''Reparametrization trick.'''

    def forward(self, mu, sigma_param):
        if self.sampling: # sample around the mean
            sigma = self.sigma(sigma_param)
            y = reparametrize(mu, sigma)
        else: # just return the mean
            y = mu
        return y

