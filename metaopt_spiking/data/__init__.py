from .cifar import cifar10, cifar100
from .mnist import mnist
from .randman import metarandmanreg
from .randman import RandomManifold
from .loader import create_dataloader, create_multitask_loader, tensors_to_loader

__all__ = [
    "cifar10",
    "cifar100",
    "create_dataloader",
    "create_multitask_loader",
    "mnist",
    "metarandmanreg",
    "RandomManifold",
    "tensors_to_loader"
]


def get_dataloader(name, **kwargs):
    """
    Return train, (valid) and test dataloader given name of dataset
    """
    if name == "cifar10":
        return cifar10(**kwargs)
    elif name == "cifar100":
        return cifar100(**kwargs)
    elif name == "mnist":
        return mnist(**kwargs)
    elif name == "sinusoid":
        return metarandmanreg(**kwargs)
    else:
        raise ValueError("Dataset \"{}\" undefined".format(name))
