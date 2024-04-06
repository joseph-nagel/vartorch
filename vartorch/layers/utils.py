'''Model layer utils.'''

from inspect import isfunction, isclass

import torch.nn as nn

from .linear import VarLinear


ACTIVATIONS = {
    'none': None,
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'swish': nn.SiLU
}


def make_activation(mode='leaky_relu', **kwargs):
    '''Create activation function.'''

    if mode is None:
        activ = None

    elif isfunction(mode):
        activ = mode

    elif isclass(mode):
        activ = mode(**kwargs)

    elif mode in ACTIVATIONS.keys():
        activ = ACTIVATIONS[mode](**kwargs)

    else:
        raise ValueError(f'Unknown activation: {mode}')

    return activ


def make_block(layers):
    '''Assemble a block of layers.'''

    if isinstance(layers, nn.Module):
        block = layers

    elif isinstance(layers, (list, tuple)):

        layers = [l for l in layers if l is not None]

        if len(layers) == 1:
            block = layers[0]
        else:
            block = nn.Sequential(*layers)

    else:
        raise TypeError(f'Invalid layers type: {type(layers)}')

    return block


def make_dropout(drop_rate=None):
    '''Create a dropout layer.'''

    if drop_rate is None:
        dropout = None
    else:
        dropout = nn.Dropout(p=drop_rate)

    return dropout


def make_dense(in_features,
               out_features,
               bias=True,
               batchnorm=False,
               activation=None,
               drop_rate=None,
               variational=False,
               var_opts={}):
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
    activation : None or str
        Nonlinearity type.
    drop_rate : float
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
            bias=bias # the bias should be disabled if a batchnorm directly follows after the convolution
        )

    # create linear variational layer
    else:
        linear = VarLinear(
            in_features,
            out_features,
            **var_opts
        )

    # create activation function
    activation = make_activation(activation)

    # create normalization
    norm = nn.BatchNorm1d(out_features) if batchnorm else None

    # assemble block
    layers = [dropout, linear, activation, norm] # note that the normalization follows the activation (which could be reversed of course)
    dense_block = make_block(layers)

    return dense_block


def make_conv(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding='same',
              bias=True,
              batchnorm=False,
              activation=None):
    '''
    Create convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    stride : int
        Stride parameter.
    padding : int
        Padding parameter.
    bias : bool
        Determines whether a bias is used.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : None or str
        Nonlinearity type.

    '''

    # create conv layer
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias # the bias should be disabled if a batchnorm directly follows after the convolution
    )

    # create activation function
    activation = make_activation(activation)

    # create normalization
    norm = nn.BatchNorm2d(out_channels) if batchnorm else None

    # assemble block
    layers = [conv, activation, norm] # note that the normalization follows the activation (which could be reversed of course)
    conv_block = make_block(layers)

    return conv_block


class SingleConv(nn.Sequential):
    '''Single conv. block.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=True,
                 batchnorm=False,
                 activation='leaky_relu'):

        # create conv layer
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias # the bias should be disabled if a batchnorm directly follows after the convolution
        )

        # create activation function
        activation = make_activation(activation)

        # create normalization
        norm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # assemble block
        layers = [conv, activation, norm] # note that the normalization follows the activation (which could be reversed of course)
        layers = [l for l in layers if l is not None]

        # initialize module
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    '''Double conv. blocks.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=True,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation='same',
                 normalize_last=True,
                 inout_first=True):

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

