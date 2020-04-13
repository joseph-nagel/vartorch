'''
Kullback-Leibler divergences.

Summary
-------
This module provides different implementations of the KL divergence.
Only diagonal Gaussian distributions are supported at the moment.

The function 'kl_div_pytorch' calculates the KL divergence
based on the native PyTorch distribution implementation.
It assumes Gaussians with arbitrary diagonal covariances.

Similarly, 'kl_div_analytical' is an analytical version for the case
that the covariance of the prior distribution is an identity matrix.

A single-sample Monte Carlo estimate can be computed 'kl_div_montecarlo'.
It can be used in conjunction with black-box variational inference schemes.

'''

import torch
import torch.distributions as dist

def kl_div_pytorch(q_mu, q_rho, p_std=1):
    '''Compute the KL divergence with PyTorch distributions.'''
    q_std = torch.log(1 + torch.exp(q_rho))
    q = dist.Normal(q_mu, q_std) # variational distribution
    p = dist.Normal(0, p_std) # prior distribution
    kl = dist.kl_divergence(q, p).sum()
    return kl

def kl_div_analytical(q_mu, q_rho):
    '''Compute KL divergence analytically.'''
    q_std = torch.log(1 + torch.exp(q_rho))
    kl = 0.5 * torch.sum(q_std**2 + q_mu**2 - torch.log(q_std**2) - 1)
    return kl

def kl_div_montecarlo(z, q_mu, q_rho, p_std=1):
    '''Compute KL divergence with a single MC sample.'''
    q_std = torch.log(1 + torch.exp(q_rho))
    log_q = dist.Normal(q_mu, q_std).log_prob(z)
    log_p = dist.Normal(0, p_std).log_prob(z)
    return (log_q - log_p).sum()

