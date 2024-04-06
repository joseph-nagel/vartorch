'''Reparametrization trick.'''

from ..reparam import reparametrize
from .base import VarLayer


class Reparametrize(VarLayer):
    '''Reparametrization trick.'''

    def forward(self, mu, sigma_param):

        # sample around the mean
        if self.sampling:
            sigma = self.sigma(sigma_param)
            y = reparametrize(mu, sigma)

        # just return the mean
        else:
            y = mu

        return y

