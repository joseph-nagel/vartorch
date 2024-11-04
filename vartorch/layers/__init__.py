'''
Variational layers.

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
    IntOrInts,
    ActivType,
    make_activation,
    make_block,
    make_dropout,
    make_dense,
    make_conv,
    SingleConv,
    DoubleConv
)

