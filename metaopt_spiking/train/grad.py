"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from utils import config


def grad_hyperparams(meta_model, train_loader, energy_function, beta=0.0, valid_loader=None, cost_function=None):
    """
    Compute hyperparameter gradients on a batch of the train_loader
    (and valid_loader) for the given augmented loss function.

    Args:
        meta_model: meta.MetaModule
        train_loader: DataLoader that contains the training data
        energy_function: energy.EnergyFunction for the free phase
        beta: Nudging strength
        valid_loader: DataLoader that contains the validation data
        cost_function: energy.EnergyFunction for the nudged phase

    Returns:
        List of gradients wrt hyperparams, gradient norm of parameters
    """
    # Prepare model for training TODO: Is this correct?
    meta_model.train(mode=True)

    # Sample training batch
    x_batch_train, y_batch_train = next(iter(train_loader))
    x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

    # Energy
    output, state = meta_model(x_batch_train)
    energy = energy_function(output, y_batch_train, meta_model, state)

    if beta > 0.0:
        # Sample validation batch
        x_batch_valid, y_batch_valid = next(iter(valid_loader))
        x_batch_valid, y_batch_valid = x_batch_valid.to(config.device), y_batch_valid.to(config.device)

        # Cost
        output, state = meta_model(x_batch_valid)
        cost = cost_function(output, y_batch_valid, meta_model, state)
    else:
        cost = 0.0

    # Total energy
    total_energy = energy + beta * cost

    return torch.autograd.grad(total_energy, meta_model.meta_parameters())
