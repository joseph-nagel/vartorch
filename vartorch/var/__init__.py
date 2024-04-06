'''
Variational mdoels.

Summary
-------
The core class 'VariationalClassification' implements a scheme
for stochastic variational inference in classification problems.
It turns models with variational and other layers into Bayesian classifiers.
While the likelihood determines whether the problem is binary or multi-class,
the model layers determine the unknown weights and their treatment.

A posterior over the weights of the variational layers is computed by
maximizing the ELBO w.r.t. the parameters of the variational distribution.
Other non-Bayesian unknown parameters of the prior and the likelihood,
such as the weights of non-variational layers, can also be learned this way.

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

