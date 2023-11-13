'''Visualization tools.'''

import numpy as np
import matplotlib.pyplot as plt


def plot_data_2d(X,
                 y,
                 labels=None,
                 colors=None,
                 ax=None):
    '''Plot data points with labels on a two-dim. plane.'''

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if labels is None:
        labels = np.unique(y)

    for idx, label in enumerate(labels):
        ax.scatter(
            X[y==label, 0],
            X[y==label, 1],
            color=None if colors is None else colors[idx],
            alpha=0.7,
            edgecolors='none',
            label=f'y={label}'
        )

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    if fig is None:
        return ax
    else:
        return fig, ax


def plot_function_2d(function,
                     levels=(0.1, 0.3, 0.5, 0.7, 0.9),
                     x_limits=None,
                     y_limits=None,
                     colorbar=True,
                     ax=None):
    '''Plot a function of two features on the plane.'''

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if x_limits is None:
        x_limits = ax.get_xlim()

    if y_limits is None:
        y_limits = ax.get_ylim()

    # create inputs
    x_values = np.linspace(*x_limits, num=201)
    y_values = np.linspace(*y_limits, num=201)

    (x_grid, y_grid) = np.meshgrid(x_values, y_values)

    xy_values = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)

    # compute outputs
    z_values = function(xy_values)

    z_grid = z_values.reshape(x_grid.shape)

    # plot function
    im1 = ax.imshow(
        z_grid,
        origin='lower',
        extent=(*x_limits, *y_limits),
        interpolation='bicubic',
        cmap='Greys',
        alpha=0.4
    )

    im2 = ax.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=levels,
        colors='black',
        alpha=0.6
    )

    if colorbar:
        plt.colorbar(im1)

    plt.clabel(im2, fmt='%1.2f')

    if fig is None:
        return ax
    else:
        return fig, ax

