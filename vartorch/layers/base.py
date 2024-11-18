'''
Variational base layer.

Summary
-------
A base class for variational layers is defined in 'VarLayer'.
This manages the main settings regarding the probabilistic weights.
In particular, it comes with a parametrization mode and a sampling switch.

'''

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..reparam import (
    sigma_from_log,
    sigma_from_rho,
    log_from_sigma,
    rho_from_sigma
)


class VarLayer(nn.Module, ABC):
    '''
    Variational layer base class.

    Summary
    -------
    This should act as the parent class for all variational layers.
    It manages how standard deviations of the variational distribution are parametrized
    and allows for turning random weight sampling on and off for the predictions.

    Parameters
    ----------
    param_mode : {'log', 'rho'}
        Parametrization type, i.e. how the non-negative standard
        deviation is represented in terms of a real-valued parameter.

    '''

    def __init__(self, param_mode: str = 'log') -> None:

        super().__init__()

        # set parametrization
        self.parametrization = param_mode

        # set initial sampling mode
        self.sampling = True

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def parametrization(self) -> str:
        '''Get parametrization mode.'''
        return self._parametrization

    @parametrization.setter
    def parametrization(self, param_mode: str) -> None:
        '''Set parametrization mode.'''

        # set parametrization type
        if param_mode in ('log', 'rho'):
            self._parametrization = param_mode
        else:
            raise ValueError(f'Unknown parametrization: {param_mode}')

        # set parametrization functions
        if param_mode == 'log':
            self.sigma = sigma_from_log
            self.sigma_inverse = log_from_sigma

        elif param_mode == 'rho':
            self.sigma = sigma_from_rho
            self.sigma_inverse = rho_from_sigma

    @property
    def sampling(self) -> bool:
        '''Get sampling mode.'''
        return self._sampling

    @sampling.setter
    def sampling(self, sample_mode: bool):
        '''Set sampling mode.'''
        self._sampling = sample_mode

