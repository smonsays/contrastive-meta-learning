"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math

import numpy as np
import torch

from .base import Classification, MetaClassification, MetaRegression
from .loader import create_dataloader, create_train_val_test_loader, create_metamultitask_loader


class RandomManifold:
    """
    Smooth random manifold built from Fourier basis.
    Based on https://github.com/fzenke/randman (MIT License, (c) Friedemann Zenke)
    """
    def __init__(self, embedding_dim, manifold_dim, smoothness, f_cutoff, amplitude_range=(1.0, 1.0),
                 frequency_range=(1.0, 1.0), phase_range=(0.0, 1.0), support_range=(0.0, 1.0), standardize=True,):
        """
        Args:
            embedding_dim : The embedding space dimension
            manifold_dim : The manifold dimension
            smoothness : The power spectrum fall-off exponenent. Determines the smoothenss of the manifold
            standardize: If True, mainfold is standardize to be in a [0,1] hypercube in embedding space
        """
        self.embedding_dim = embedding_dim
        self.manifold_dim = manifold_dim
        self.smoothness = smoothness

        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range
        self.phase_range = phase_range
        self.support_range = support_range

        self.standardize = standardize

        # Determine largest frequency with non-zero weighting (given max numerical precision)
        self.f_cutoff = int(min((math.ceil(torch.finfo(torch.float).eps**(-1 / self.smoothness)), f_cutoff)))
        self.spectral_coeffs = 1.0 / (torch.arange(1, self.f_cutoff + 1)**self.smoothness)

        # Sample the parameters
        self.reset_parameters()

    def __call__(self, x):
        """
        Compute function values of the manifold in embedding space given a point x in intrinsic manifold coordinates.
        Fourier component calculation is parallelised.
        """
        components = torch.zeros(len(x), self.embedding_dim, self.manifold_dim, self.f_cutoff)
        for k in range(self.f_cutoff):
            components[:, :, :, k] = self.spectral_coeffs[k] * self.amplitude[:, :, k] *  \
                torch.sin(2 * math.pi * ((k + 1) * torch.einsum("bi, ji -> bji", x, self.frequency[:, :, k]) + self.phase[:, :, k]))

        return torch.prod(torch.sum(components, dim=-1), dim=-1)

    def _sample(self, num_samples):
        """
        Sample random points from the manifold.
        """
        lower, upper = self.support_range
        x = ((upper - lower) * torch.rand(num_samples, self.manifold_dim)) + lower

        return x, self(x)

    def reset_parameters(self):
        for key in ["amplitude", "frequency", "phase"]:
            lower, upper = self.__getattribute__(key + "_range")
            param = ((upper - lower) * torch.rand(self.embedding_dim, self.manifold_dim, self.f_cutoff)) + lower

            self.__setattr__(key, param)

    def sample(self, num_samples):
        """
        Sample random points from the standardised manifold.
        """
        x, y = self._sample(num_samples)

        if self.standardize:
            y_min, y_max = self.min_max
            y = (y - y_min) / (y_max - y_min)

        return x, y

    @property
    def min_max(self):
        """
        Estimate the minimum and maximum values from samples.
        """
        _, y = self._sample(1000)

        return (torch.amin(y, dim=0), torch.amax(y, dim=0))


def metarandman_dataset(num_way, num_shot, num_test, num_tasks, embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):

    # Create random manifolds
    manifolds = [
        RandomManifold(embedding_dim, manifold_dim, smoothness, f_cutoff)
        for _ in range(num_tasks * num_way)
    ]

    return MetaClassification(embedding_dim + manifold_dim, manifolds, num_tasks, num_way, num_shot, num_test, spiking=spiking)


def metarandman(meta_batch_size, num_batches, num_way, num_shot, num_test, num_tasks_train, num_tasks_valid,
                num_tasks_test, embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):

    # Create a metarandman train and test set
    meta_train_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_train, embedding_dim,
                                         manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_valid_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_valid, embedding_dim,
                                         manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_test_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_test, embedding_dim,
                                        manifold_dim, smoothness, f_cutoff, spiking=spiking)

    # Create dataloader that sample (with replacement for training) from the meta tasks
    meta_train_loader = create_dataloader(meta_train_set, None, meta_batch_size, sampler=torch.utils.data.RandomSampler(
        meta_train_set, replacement=True, num_samples=num_batches * meta_batch_size))

    meta_valid_loader = create_dataloader(meta_valid_set, shuffle=False, batch_size=1)
    meta_test_loader = create_dataloader(meta_test_set, shuffle=False, batch_size=1)

    return meta_train_loader, meta_valid_loader, meta_test_loader


def metarandman_multi(batch_size, num_way, num_shot, num_test, num_tasks_train, num_tasks_valid, num_tasks_test,
                      embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):
    """
    Convert metarandman into multitask dataloader.
    """
    # Create a metarandman train and test set
    meta_train_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_train, embedding_dim,
                                         manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_valid_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_valid, embedding_dim, manifold_dim,
                                         smoothness, f_cutoff, spiking=spiking)

    meta_test_set = metarandman_dataset(num_way, num_shot, num_test, num_tasks_test, embedding_dim, manifold_dim,
                                        smoothness, f_cutoff, spiking=spiking)

    return create_metamultitask_loader(meta_train_set, meta_valid_set, meta_test_set, batch_size)


def metarandman_single(batch_size, num_way, num_shot, num_test, embedding_dim, manifold_dim, smoothness,
                       f_cutoff, spiking=False, validation_split=None, train_subset=None):
    """
    Convenience method to extract a single task from metarandman.
    """
    # Create a metarandman dataset with a single task
    meta_dataset = metarandman_dataset(num_way, num_shot, num_test, 1, embedding_dim,
                                       manifold_dim, smoothness, f_cutoff, spiking=spiking)

    (x_train, y_train), (x_test, y_test) = meta_dataset[0]

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    return create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split, train_subset)


def metarandmanreg_dataset(num_shot, num_test, num_tasks, embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):

    # Create random manifolds
    # NOTE: Divide frequency by 1 / 2*pi to mimic original sinusoid regression task
    manifolds = [
        RandomManifold(
            embedding_dim, manifold_dim, smoothness, f_cutoff, amplitude_range=(0.1, 5.0),
            frequency_range=(1.0 / (2 * np.pi), 1.0 / (2 * np.pi)), phase_range=(0.0, 0.5),
            support_range=(-5.0, 5.0), standardize=False
        )
        for _ in range(num_tasks)
    ]

    if not spiking:
        dataset = MetaRegression(manifold_dim, embedding_dim, manifolds, num_shot, num_test, spiking=False)
    else:
        # In the spiking case the `input_dim` specifies the number of neurons used for the population code
        dataset = MetaRegression(100, embedding_dim, manifolds, num_shot, num_test, spiking=True)

    return dataset


def metarandmanreg(meta_batch_size, num_batches, num_shot, num_test, num_tasks_valid, num_tasks_test,
                   embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):

    # Create a metarandman train and test set
    meta_train_set = metarandmanreg_dataset(num_shot, num_test, num_batches * meta_batch_size, embedding_dim,
                                            manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_valid_set = metarandmanreg_dataset(num_shot, num_test, num_tasks_valid, embedding_dim,
                                            manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_test_set = metarandmanreg_dataset(num_shot, num_test, num_tasks_test, embedding_dim,
                                           manifold_dim, smoothness, f_cutoff, spiking=spiking)

    # Create dataloader that sample (with replacement for training) from the meta tasks
    meta_train_loader = create_dataloader(meta_train_set, shuffle=True, batch_size=meta_batch_size)
    meta_valid_loader = create_dataloader(meta_valid_set, shuffle=False, batch_size=1)
    meta_test_loader = create_dataloader(meta_test_set, shuffle=False, batch_size=1)

    return meta_train_loader, meta_valid_loader, meta_test_loader


def metarandmanreg_multi(batch_size, num_shot, num_test, num_tasks_train, num_tasks_valid, num_tasks_test,
                         embedding_dim, manifold_dim, smoothness, f_cutoff, spiking=False):
    """
    Convert metarandman into multitask dataloader.
    """
    # Create a metarandman train and test set
    meta_train_set = metarandmanreg_dataset(num_shot, num_test, num_tasks_train, embedding_dim,
                                            manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_valid_set = metarandmanreg_dataset(num_shot, num_test, num_tasks_valid, embedding_dim,
                                            manifold_dim, smoothness, f_cutoff, spiking=spiking)

    meta_test_set = metarandmanreg_dataset(num_shot, num_test, num_tasks_test, embedding_dim,
                                           manifold_dim, smoothness, f_cutoff, spiking=spiking)

    return create_metamultitask_loader(meta_train_set, meta_valid_set, meta_test_set, batch_size)


def metarandmanreg_single(batch_size, num_shot, num_test, embedding_dim, manifold_dim, smoothness,
                          f_cutoff, spiking=False, validation_split=None, train_subset=None):
    """
    Convenience method to extract a single task from metarandman.
    """
    # Create a metarandman dataset with a single task
    meta_dataset = metarandmanreg_dataset(num_shot, num_test, 1, embedding_dim, manifold_dim,
                                          smoothness, f_cutoff, spiking=spiking)

    (x_train, y_train), (x_test, y_test) = meta_dataset[0]

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    return create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split, train_subset)


def randman_dataset(num_classes, num_samples_per_class, embedding_dim, manifold_dim, smoothness, f_cutoff, spiking):
    # Create one random manifold per class
    manifolds = [
        RandomManifold(embedding_dim, manifold_dim, smoothness, f_cutoff)
        for c in range(num_classes)
    ]

    return Classification(embedding_dim, manifolds, num_samples_per_class, spiking)


def randman(batch_size, validation_split=None, train_subset=None, spiking=False):
    """
    Create train-, (valid-) and testloader for Randman.

    Args:
        batch_size: Number of training samples per batch
        validation_split: Float defining the fraction of training data split into a validation set

    Returns:
        train_loader: Training torch.utils.data.DataLoader
        valid_loader: Validation torch.utils.data.DataLoader
        test_loader: Test torch.utils.data.DataLoader
    """
    full_dataset = randman_dataset(num_classes=5, num_samples_per_class=100 + 100, embedding_dim=20,
                                   manifold_dim=3, smoothness=1, f_cutoff=10, spiking=spiking)

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [500, 500])

    return create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split, train_subset)
