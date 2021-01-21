'''
Reparametrization trick.

Summary
-------
The reparametrization trick from variational autoencoders is used for the MC simulation.
It transforms the expectation over the variational posterior
into an expectation over a standardized distribution.
Consequentially, one can exchange the gradient and the expectation operator.
The gradient of an expectation has been eventually expressed as an expectation of a gradient,
which allows for MC sampling to estimate of the gradient with backpropagation.

In this module, the reparametrization trick is implemented as a subclass
'Reparametrize' of PyTorch's nn.Module and as the function 'reparametrize'.

'''

import torch
import torch.nn as nn

class Reparametrize(nn.Module):
    '''Class performing the reparametrization trick.'''

    def __init__(self):
        super().__init__()
        self.sampling = True

    @property
    def sampling(self):
        '''Get sampling mode.'''
        return self._sampling

    @sampling.setter
    def sampling(self, mode):
        '''Set sampling mode.'''
        self._sampling = mode

    def forward(self, mu, rho):
        if self.sampling: # sample around the mean
            y = reparametrize(mu, rho)
        else: # just return the mean
            y = mu
        return y

def reparametrize(mu, rho):
    '''Perform the reparametrization trick.'''
    sigma = torch.log(1 + torch.exp(rho))
    eps = torch.randn_like(sigma)
    return mu + (eps * sigma)

