'''
Variational layers.

Summary
-------
A base class for variational layers is defined in 'VarLayer'.
This manages the main settings regarding the probabilistic weights.
In particular, it comes with a parametrization mode and a sampling switch.

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

'Reparametrize' implements the reparametrization trick for the MC estimation of the ELBO.
It can be used as component for building advanced layers.

Modules
-------
base : Variational base layer.
linear : Variational linear layers.
reparam : Reparametrization trick.
utils : Model layer utils.

'''

from . import (
    base,
    linear,
    reparam,
    utils
)

from .base import VarLayer

from .linear import (
    VarLinear,
    VarLinearWithUncertainLogits,
    VarLinearWithLearnableTemperature
)

from .reparam import Reparametrize

from .utils import (
    make_activation,
    make_block,
    make_dropout,
    make_dense,
    make_conv
)

