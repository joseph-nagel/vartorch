'''
Datamodules.

Modules
-------
mnist : MNIST datamodule.
moons : Half-moons datamodule.

'''

from . import mnist, moons
from .mnist import MNISTDataModule
from .moons import make_half_moons, MoonsDataModule
