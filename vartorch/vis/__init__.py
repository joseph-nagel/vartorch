'''
Visualization tools.

Modules
-------
mnist : For the MNIST example.
moons : For the half-moons example.

'''

from . import mnist, moons


from .mnist import (
    plot_point_predictions,
    plot_post_predictions,
    plot_entropy_histograms
)

from .moons import plot_data_2d, plot_function_2d

