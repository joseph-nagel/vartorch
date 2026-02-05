'''Model layer utils.'''

from typing import Any
from collections.abc import Sequence
from inspect import isclass

import torch.nn as nn

from .linear import VarLinear


IntOrInts = int | tuple[int, int]
ActivType = str | type[nn.Module]


ACTIVATIONS = {
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'swish': nn.SiLU
}


def make_activation(mode: ActivType | None = 'leaky_relu', **kwargs: Any) -> nn.Module | None:
    '''Create activation function.'''
    if mode is None:
        activ = None
    elif isclass(mode):
        activ = mode(**kwargs)
    elif isinstance(mode, str) and mode in ACTIVATIONS.keys():
        activ = ACTIVATIONS[mode](**kwargs)
    else:
        raise ValueError(f'Unknown activation: {mode}')
    return activ


def make_block(layers: nn.Module | Sequence[nn.Module | None]) -> nn.Module:
    '''Assemble a block of layers.'''
    if isinstance(layers, nn.Module):
        block = layers
    elif isinstance(layers, (list, tuple)):
        not_none_layers = [l for l in layers if l is not None]
        if len(not_none_layers) == 0:
            raise ValueError('No layers to assemble')
        elif len(not_none_layers) == 1:
            block = not_none_layers[0]
        else:
            block = nn.Sequential(*not_none_layers)
    else:
        raise TypeError(f'Invalid layers type: {type(layers)}')
    return block


def make_dropout(drop_rate: float | None = None) -> nn.Module | None:
    '''Create a dropout layer.'''
    if drop_rate is None:
        dropout = None
    else:
        dropout = nn.Dropout(p=drop_rate)
    return dropout


def make_dense(
    in_features: int,
    out_features: int,
    bias: bool = True,
    batchnorm: bool = False,
    activation: ActivType | None = 'leaky_relu',
    drop_rate: float | None = None,
    variational: bool = False,
    var_opts: dict[str, Any] = {}
) -> nn.Module:
    '''
    Create fully connected layer.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    bias : bool
        Determines whether a bias is used.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str or None
        Nonlinearity type.
    drop_rate : float or None
        Dropout probability.
    variational : bool
        Determines whether layer is variational.
    var_opts : dict
        Options for variational linear layer.

    '''

    # create dropout layer
    dropout = make_dropout(drop_rate=drop_rate)

    # create linear layer
    if not variational:
        linear = nn.Linear(
            in_features,
            out_features,
            bias=bias  # the bias should be disabled if a batchnorm directly follows after the linear layer
        )

    # create linear variational layer
    else:
        linear = VarLinear(
            in_features,
            out_features,
            **var_opts
        )

    # create activation function
    activ = make_activation(activation)

    # create normalization
    norm = nn.BatchNorm1d(out_features) if batchnorm else None

    # assemble block
    layers = [dropout, linear, activ, norm]  # note that the normalization follows the activation (which could be reversed of course)
    dense_block = make_block(layers)

    return dense_block


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: IntOrInts = 3,
    stride: IntOrInts = 1,
    padding: IntOrInts | str = 'same',
    bias: bool = True,
    batchnorm: bool = False,
    activation: ActivType | None = 'leaky_relu'
) -> nn.Module:
    '''
    Create convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or (int, int)
        Convolutional kernel size.
    stride : int or (int, int)
        Stride parameter.
    padding : int, (int, int) or str
        Padding parameter.
    bias : bool
        Determines whether a bias is used.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str or None
        Nonlinearity type.

    '''

    # create conv layer
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias  # the bias should be disabled if a batchnorm directly follows after the convolution
    )

    # create activation function
    activ = make_activation(activation)

    # create normalization
    norm = nn.BatchNorm2d(out_channels) if batchnorm else None

    # assemble block
    layers = [conv, activ, norm]  # note that the normalization follows the activation (which could be reversed of course)
    conv_block = make_block(layers)

    return conv_block


class SingleConv(nn.Sequential):
    '''Single conv. block.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrInts = 3,
        stride: IntOrInts = 1,
        padding: IntOrInts | str = 'same',
        bias: bool = True,
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu'
    ):

        # create conv layer
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias  # the bias should be disabled if a batchnorm directly follows after the convolution
        )

        # create activation function
        activ = make_activation(activation)

        # create normalization
        norm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # assemble block
        layers = [conv, activ, norm]  # note that the normalization follows the activation (which could be reversed of course)
        not_none_layers = [l for l in layers if l is not None]

        # initialize module
        super().__init__(*not_none_layers)


class DoubleConv(nn.Sequential):
    '''Double conv. blocks.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrInts = 3,
        stride: IntOrInts = 1,
        padding: IntOrInts | str = 'same',
        bias: bool = True,
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        normalize_last: bool = True,
        inout_first: bool = True
    ):

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # create first conv
        conv_block1 = SingleConv(
            in_channels,
            out_channels if inout_first else in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            batchnorm=batchnorm,
            activation=activation
        )

        # create second conv
        conv_block2 = SingleConv(
            out_channels if inout_first else in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            batchnorm=(batchnorm and normalize_last),
            activation=last_activation
        )

        # initialize module
        super().__init__(conv_block1, conv_block2)
