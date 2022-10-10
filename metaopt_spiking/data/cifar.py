"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from torchvision import datasets, transforms

from .base import DATAPATH
from .loader import create_train_val_test_loader


def cifar10_dataset(is_train, augmentation=False):
    """
    CIFAR10 dataset with the standard data transformation.

    Args:
        is_train: Bool indicating whether to get the train or test data

    Returns:
        dataset: CIFAR10 torch.utils.data.Dataset
    """
    transform_list = []

    if is_train and augmentation:
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                               std=(0.2023, 0.1994, 0.2010)))

    dataset = datasets.CIFAR10(DATAPATH, train=is_train,
                               transform=transforms.Compose(transform_list), download=True)

    return dataset


def cifar10(batch_size, validation_split=None, train_subset=None):
    """
    Create train-, (valid-) and testloader for CIFAR10.

    Args:
        batch_size: Number of training samples per batch
        validation_split: Float defining the fraction of training data split into a validation set

    Returns:
        train_loader: Training torch.utils.data.DataLoader
        valid_loader: Validation torch.utils.data.DataLoader
        test_loader: Test torch.utils.data.DataLoader
    """
    # Get CIFAR10 dataset
    train_dataset = cifar10_dataset(True)
    test_dataset  = cifar10_dataset(False)

    # HACK: Putting CIFAR10 on the device (GPU) will remove any augmentations
    return create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split, train_subset, on_device=True)


def cifar100_dataset(is_train, augmentation=False):
    """
    CIFAR100 dataset with the standard data transformation.

    Args:
        is_train: Bool indicating whether to get the train or test data

    Returns:
        dataset: CIFAR100 torch.utils.data.Dataset
    """
    transform_list = []

    if is_train and augmentation:
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                               std=(0.2675, 0.2565, 0.2761)))

    dataset = datasets.CIFAR100(DATAPATH, train=is_train,
                                transform=transforms.Compose(transform_list), download=True)

    return dataset


def cifar100(batch_size, validation_split=None, train_subset=None):
    """
    Create train-, (valid-) and testloader for CIFAR100.

    Args:
        batch_size: Number of training samples per batch
        validation_split: Float defining the fraction of training data split into a validation set

    Returns:
        train_loader: Training torch.utils.data.DataLoader
        valid_loader: Validation torch.utils.data.DataLoader
        test_loader: Test torch.utils.data.DataLoader
    """
    # Get CIFAR100 dataset
    train_dataset = cifar100_dataset(True)
    test_dataset  = cifar100_dataset(False)

    return create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split, train_subset)
