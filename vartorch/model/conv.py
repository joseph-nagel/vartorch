'''Conv. models.'''

from collections.abc import Sequence

import torch.nn as nn

from ..layers import (
    IntOrInts,
    ActivType,
    SingleConv,
    DoubleConv
)


class ConvDown(nn.Sequential):
    '''Convolutions with downsampling.'''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: IntOrInts = 3,
        padding: IntOrInts | str = 'same',
        stride: IntOrInts = 1,
        pooling: IntOrInts | None = 2,
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        normalize_last: bool = True,
        pool_last: bool = True,
        double_conv: bool = False,
        inout_first: bool = True
    ) -> None:

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

        # create specific options
        kwargs = {'inout_first': inout_first} if double_conv else {}

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_channels) >= 2:
            num_layers = len(num_channels) - 1
        else:
            raise ValueError('Number of channels needs at least two entries')

        # assemble layers
        layers = []  # type: list[nn.Module]

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_last = (idx < num_layers - 1)

            # create conv layer
            conv = ConvType(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                activation=activation if is_not_last else last_activation,
                **kwargs
            )

            layers.append(conv)

            # create pooling layer
            if pooling is not None:
                if is_not_last or pool_last:
                    down = nn.MaxPool2d(pooling)

                    layers.append(down)

        # initialize module
        super().__init__(*layers)

