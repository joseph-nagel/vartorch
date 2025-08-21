'''
Reparametrization issues.

Summary
-------
Two different parametrization issues are settled in this module.
First, the reparametrization trick from variational autoencoders is implemented in `reparametrize`.
During the MC simulation of the loss, it transforms the expectation over the
variational posterior into an expectation over a standardized distribution.
Consequentially, one can exchange the gradient and the expectation operator.
The gradient of an expectation has been eventually expressed as an expectation of a gradient,
which allows for MC sampling to estimate of the gradient with backpropagation.

A second parametrization issue emerges for the standard deviation of the variational distribution.
While it is non-negative, it should be represented by a real-valued and possibly negative parameter.
Two alternative parametrizations are therefore supported through `sigma_from_log` and `sigma_from_rho`.
Here, the standard deviation is calculated from real-valued parameters via simple transformations.

'''

import torch


def reparametrize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    '''Sample with the reparametrization trick.'''
    eps = torch.randn_like(sigma)
    return mu + (eps * sigma)


def sigma_from_log(sigma_param: torch.Tensor) -> torch.Tensor:
    '''Calculate sigma from log-sigma.'''
    log_sigma = sigma_param
    sigma = torch.exp(log_sigma)
    return sigma


def sigma_from_rho(sigma_param: torch.Tensor) -> torch.Tensor:
    '''Calculate sigma from rho.'''
    rho = sigma_param
    sigma = torch.log(1 + torch.exp(rho))
    return sigma


def log_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    '''Calculate log-sigma from sigma.'''
    log_sigma = torch.log(sigma)
    return log_sigma


def rho_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    '''Calculate rho from sigma.'''
    rho = torch.log(torch.exp(sigma) - 1)
    return rho

