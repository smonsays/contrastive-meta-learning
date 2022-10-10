"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from utils import config, first_elem

from .base import extended_dataset
from .utils import split_dataset


def create_dataloader(dataset, shuffle, batch_size, sampler=None, num_workers=0, persistent_workers=False, on_device=False):
    """
    Create a dataloader for a given dataset.

    Args:
        dataset: torch.utils.data.Dataset used in the dataloader
        shuffle: Bool indicating whether to shuffle the data
        batch_size: Number of training samples per batch
        sampler: Optional torch.utils.data.Sampler
        num_workers: Number of subprocesses to use for data loading (0=main process).
    Returns:
        torch.utils.data.DataLoader
    """
    # For GPU acceleration store dataloader in pinned (page-locked) memory
    pin_memory = True if torch.cuda.is_available() else False

    if on_device:
        # Move the dataset onto the device (GPU)
        imgs, labels = [], []
        for img, label in dataset:
            imgs.append(img)
            labels.append(label)

        imgs = torch.stack(imgs).to(config.device)
        labels = torch.tensor(labels).to(config.device)
        dataset = torch.utils.data.TensorDataset(imgs, labels)

    if first_elem(dataset).device.type != "cpu":
        # Set GPU compatible options
        num_workers = 0
        persistent_workers = False
        pin_memory = False

    # Create dataloader objects
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler,
                                             shuffle=shuffle, drop_last=False,
                                             num_workers=num_workers, pin_memory=pin_memory,
                                             persistent_workers=persistent_workers)
    return dataloader


def create_metamultitask_loader(meta_train_set, meta_valid_set, meta_test_set, batch_size):
    # Create a single training dataset from all meta-train tasks (train + valid data)
    train_set_x, train_set_y, train_set_task = [], [], []
    for task, ((x_train, y_train), (x_valid, y_valid)) in enumerate(meta_train_set):
        train_set_x.append(torch.cat([x_train, x_valid]))
        train_set_y.append(torch.cat([y_train, y_valid]))
        train_set_task.append(task * torch.ones(len(x_train) + len(x_valid)))

    # Combine the task-id into the input tensor to keep the dataloader compatible with existing abstractions
    train_set_x_task = torch.cat([torch.cat(train_set_x), torch.cat(train_set_task).unsqueeze(1)], dim=1)
    train_dataset = torch.utils.data.TensorDataset(train_set_x_task, torch.cat(train_set_y))

    # Monkey patch TensorDataset to adhere to ExtendedDataset interface
    train_dataset = extended_dataset(
        train_dataset, input_dim=meta_train_set.input_dim, output_dim=meta_train_set.output_dim, type=meta_train_set.type
    )

    # Create standard train dataloader and meta test dataloader
    train_loader = create_dataloader(train_dataset, shuffle=True, batch_size=batch_size)

    meta_valid_loader = create_dataloader(meta_valid_set, shuffle=False, batch_size=1)
    meta_test_loader = create_dataloader(meta_test_set, shuffle=False, batch_size=1)

    return train_loader, meta_valid_loader, meta_test_loader


def create_multitask_loader(datasets, batch_size, train_subset=None):
    """
    Create multitask loader.

    Args:
        datasets: list of torch.utils.data.Dataset for training
        batch_size: Number of training samples per batch
        train_subset: Number of training subsamples
    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    if train_subset is not None:
        datasets = [torch.utils.data.Subset(d, range(train_subset)) for d in datasets]

    # Combine all train datasets into a single big data set
    dataset_all = torch.utils.data.ConcatDataset(datasets)

    # Create a single big train loader containing all the tasks
    dataloader = create_dataloader(dataset_all, True, batch_size)

    return dataloader


def create_train_val_test_loader(train_dataset, test_dataset, batch_size, validation_split=None, train_subset=None, on_device=False):
    """
    Create train-, validation- and test loader from a given train- and test dataset.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Number of training samples per batch
        validation_split: Float defining the fraction of training data split into a validation set
        train_subset: Number of training subsamples

    Returns:
        train_loader: Training torch.utils.data.DataLoader
        valid_loader: Validation torch.utils.data.DataLoader
        test_loader: Test torch.utils.data.DataLoader
    """
    if train_subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset))

    if validation_split is None:
        # Create dataloaders
        train_loader = create_dataloader(train_dataset, True, batch_size, on_device=on_device)
        test_loader  = create_dataloader(test_dataset, False, batch_size, on_device=on_device)

        return train_loader, None, test_loader

    else:
        # Split the training set into training and validation set
        train_dataset, valid_dataset = split_dataset(train_dataset, validation_split)

        # Create dataloaders
        train_loader = create_dataloader(train_dataset, True, batch_size, on_device=on_device)
        valid_loader = create_dataloader(valid_dataset, False, batch_size, on_device=on_device)
        test_loader  = create_dataloader(test_dataset, False, batch_size, on_device=on_device)

        return train_loader, valid_loader, test_loader


def tensors_to_loader(tensors, shuffle, batch_size):
    """
    Create dataloader from list of tensors.
    """
    if batch_size >= len(tensors[0]):
        return [tensors]
    else:
        dataset = torch.utils.data.TensorDataset(*tensors)
        return create_dataloader(dataset, shuffle, batch_size)
