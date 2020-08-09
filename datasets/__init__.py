from __future__ import absolute_import

from .cifar import *
from .mnist import *
from .imagenet import *

__factory = {
    'cifar10': get_cifar10,
    'imagenet': get_imagenet,
    'mnist': get_mnist
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a train dataset loader, a val dataset loader and a class number.

    Parameters
    ----------
    name : str
        Dataset name. Can be one of 'cifar10', 'mnist' and 'imagenet'
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)
