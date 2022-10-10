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


def mnist_dataset(is_train):
    """
    Get the MNIST dataset with the standard data transformation.

    Args:
        is_train: Bool indicating whether to get the train or test data

    Returns:
        dataset: MNIST torch.utils.data.Dataset
    """
    dataset = datasets.MNIST(DATAPATH, train=is_train, download=True, transform=transforms.Compose([
                             # Could insert transforms.Pad(2) here to obtain dimensions 32x32
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                             ]))
    return dataset


def mnist(batch_size, validation_split=None, train_subset=None):
    """
    Create train-, (valid-) and testloader for MNIST.

    Args:
        batch_size: Number of training samples per batch
        validation_split: Float defining the fraction of training data split into a validation set

    Returns:
        train_loader: Training torch.utils.data.DataLoader
        valid_loader: Validation torch.utils.data.DataLoader
        test_loader: Test torch.utils.data.DataLoader
    """
    # Get MNIST dataset
    train_dataset = mnist_dataset(True)
    test_dataset  = mnist_dataset(False)

    return create_train_val_test_loader(
        train_dataset, test_dataset, batch_size, validation_split, train_subset, on_device=True
    )
