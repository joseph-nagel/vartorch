'''Dense models.'''

import torch.nn as nn

from ..layers import make_dense


class DenseBlock(nn.Sequential):
    '''Multiple (serial) dense layers.'''

    def __init__(self,
                 num_features,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation=None,
                 normalize_last=False,
                 drop_rate=None,
                 variational=False,
                 var_opts={}):

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_features) >= 2:
            num_layers = len(num_features) - 1
        else:
            raise ValueError('Number of features needs at least two entries')

        # assemble layers
        layers = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = make_dense(
                in_features,
                out_features,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                activation=activation if is_not_last else last_activation,
                drop_rate=drop_rate,
                variational=variational,
                var_opts=var_opts
            )

            layers.append(dense)

        # initialize module
        super().__init__(*layers)

