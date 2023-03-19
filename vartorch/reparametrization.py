'''
Reparametrization tricks.

Summary
-------
Two different parametrization issues are settled in this module.
First, the reparametrization trick from variational autoencoders is implemented in 'reparametrize'.
During the MC simulation of the loss, it transforms the expectation over the
variational posterior into an expectation over a standardized distribution.
Consequentially, one can exchange the gradient and the expectation operator.
The gradient of an expectation has been eventually expressed as an expectation of a gradient,
which allows for MC sampling to estimate of the gradient with backpropagation.

A second parametrization issue emerges for the standard deviation of the variational distribution.
While it is non-negative, it should be represented by a real-valued and possibly negative parameter.
Two alternative parametrizations are therefore supported through 'sigma_from_log' and 'sigma_from_rho'.
Here, the standard deviation is calculated from real-valued parameters via simple transformations.

'''

import torch


def reparametrize(mu, sigma):
    '''Sample with the reparametrization trick.'''
    eps = torch.randn_like(sigma)
    return mu + (eps * sigma)


def sigma_from_log(log_sigma):
    '''Calculate sigma from log-sigma.'''
    sigma = torch.exp(log_sigma) # log_sigma = torch.log(sigma)
    return sigma


def sigma_from_rho(rho):
    '''Calculate sigma from rho.'''
    sigma = torch.log(1 + torch.exp(rho)) # rho = torch.log(torch.exp(sigma) - 1)
    return sigma

