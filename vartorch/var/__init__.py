'''
Variational models.

Modules
-------
base : Variational classifier.
conv : Conv. variational classifier.
dense : Dense variational classifier.

'''

from . import base, conv, dense


from .base import VarClassifier

from .conv import ConvVarClassifier

from .dense import DenseVarClassifier

