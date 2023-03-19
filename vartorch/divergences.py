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


def kl_div_pytorch(q_mu, q_sigma, p_sigma=1.):
    '''Compute the KL divergence with PyTorch distributions.'''
    q = dist.Normal(q_mu, q_sigma) # variational distribution
    p = dist.Normal(0, p_sigma) # prior distribution
    kl = dist.kl_divergence(q, p).sum()
    return kl


def kl_div_analytical(q_mu, q_sigma):
    '''Compute KL divergence analytically.'''
    kl = 0.5 * torch.sum(q_sigma**2 + q_mu**2 - torch.log(q_sigma**2) - 1)
    return kl


def kl_div_montecarlo(z, q_mu, q_sigma, p_sigma=1.):
    '''Compute KL divergence with a single MC sample.'''
    log_q = dist.Normal(q_mu, q_sigma).log_prob(z)
    log_p = dist.Normal(0, p_sigma).log_prob(z)
    return (log_q - log_p).sum()

