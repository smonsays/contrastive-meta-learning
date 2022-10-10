"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging

import higher
import torch

from utils import config, create_optimizer
from .utils import infinite_iterator, make_optimizer_differentiable


def train(model, num_steps, optimizer, train_loader, loss_function, verbose=False):
    """
    Train a model on the train_loader for `num_steps` batches sampled i.i.d.

    Args:
        model: torch.nn.Module
        num_steps: Number of batches to train on
        optimizer: torch.optim.Optimizer for the model parameters
        train_loader: torch.utils.data.DataLoader with the training dataset
        loss_function: energy.EnergyFunction to compute the loss
        verbose: Logging verbosity
    """
    # Convert dataloader into infinite iterator
    train_iterator = infinite_iterator(train_loader)

    # Prepare model for training
    model.train(mode=True)

    # Run the training loop over the whole dataset
    for step in range(num_steps):

        # Sample training batch
        x_batch, y_batch = next(train_iterator)
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Compute forward pass on current batch
        output, state = model(x_batch)

        # Compute batch tot_loss as cross entropy tot_loss
        batch_loss = loss_function(output, y_batch, model, state)

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan.")

        # Compute gradients wrt current batch tot_loss and perform parameter update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if verbose and (step == 0 or (step + 1) % int(num_steps / 10) == 0):
            # Compute batch accuracy
            with torch.no_grad():
                grad_norm = torch.linalg.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])).item()
                pred = torch.max(output, dim=1)[1]
                batch_acc = torch.sum(torch.eq(pred, y_batch)).item() / x_batch.size(0)
            logging.info("step: {}/{}: batch_loss: {:.4f} \t batch_acc: {:.2f} \t grad_norm: {:.2f}".format(
                step + 1, num_steps, batch_loss, batch_acc, grad_norm))


def train_augmented(meta_model, num_steps, optimizer, train_loader, energy_function,
                    beta=0.0, valid_loader=None, cost_function=None, scheduler=None, verbose=False):
    """
    Train the base_parameters of a meta_model on the augmented loss.

    Args:
        meta_model: meta.MetaModule
        num_steps: Number of relaxation steps
        optimizer: Optimizer for the base parameters
        train_loader: DataLoader that contains the training data
        energy_function: energy.EnergyFunction for the free phase
        beta: Nudging strength
        valid_loader: DataLoader that contains the validation data
        cost_function: energy.EnergyFunction for the nudged phase
        scheduler: Learning rate scheduler for `optimizer`
        verbose: Logging verbosity

    Returns:
        Gradient norm of parameters
    """
    # Convert dataloader into infinite iterator
    train_iterator = infinite_iterator(train_loader)
    if beta > 0.0:
        valid_iterator = infinite_iterator(valid_loader)

    # Prepare model for training
    meta_model.train(mode=True)

    for step in range(num_steps):
        # If network is stochastic, sample one set of parameters to be used for both the energy and cost
        try:
            meta_model.base_learner.sample()
        except AttributeError:
            pass

        # Sample training batch
        x_batch_train, y_batch_train = next(train_iterator)
        x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

        # Energy
        output_train, state_train = meta_model(x_batch_train)
        energy = energy_function(output_train, y_batch_train, meta_model, state_train)

        if beta > 0.0:
            # Sample validation batch
            x_batch_valid, y_batch_valid = next(valid_iterator)
            x_batch_valid, y_batch_valid = x_batch_valid.to(config.device), y_batch_valid.to(config.device)

            # Cost
            output_valid, state_valid = meta_model(x_batch_valid)
            cost = cost_function(output_valid, y_batch_valid, meta_model, state_valid)
        else:
            cost = 0.0

        # Total energy
        total_energy = energy + beta * cost

        if torch.isnan(total_energy):
            raise ValueError("Loss diverged to nan.")

        # Compute gradients wrt base parameters
        if hasattr(meta_model.base_learner, 'custom_grad'):
            with torch.no_grad():
                # Use custom gradient computation if implemented
                grad_params = meta_model.base_learner.custom_grad(
                    x_batch_train, y_batch_train, state_train, energy_function, meta_model.meta_learner
                )
                if beta > 0.0:
                    grad_params_cost = meta_model.base_learner.custom_grad(
                        x_batch_valid, y_batch_valid, state_valid, cost_function, meta_model.meta_learner
                    )
                    for idx, grad_cost in enumerate(grad_params_cost):
                        grad_params[idx] += beta * grad_cost
        else:
            # Default to autograd based gradient computation
            grad_params = torch.autograd.grad(total_energy, meta_model.base_parameters())

        # Update the base parameters
        optimizer.zero_grad()

        for p, p_grad in zip(meta_model.base_parameters(), grad_params):
            p.grad = p_grad

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Logging
        if verbose and (step == 0 or (step + 1) % int(num_steps / 10) == 0):
            # HACK: Keeping track of global time by storing a variable in config.writer
            try:
                config.writer.relaxation_step += 1
            except AttributeError:
                config.writer.relaxation_step = 0
            config.writer.add_scalars('relaxation', {'energy': energy, 'cost': cost}, config.writer.relaxation_step)

            with torch.no_grad():
                grad_norm = torch.linalg.norm(torch.cat([p.grad.view(-1) for p in meta_model.base_parameters()]))

            logging.info("step: {}/{} \t total_energy: {:4f} \t energy: {:4f} \t cost: {:4f} \t grad_norm: {:4f}".format(
                step + 1, num_steps, total_energy, energy, cost, grad_norm))

    # Compute the gradient norm as a proxy for how close we got to a fixed point
    grad_norm = torch.linalg.norm(torch.cat([p.grad.view(-1) for p in meta_model.base_parameters()]))

    return grad_norm


def train_differentiable(model, num_steps, optimizer_name, lr, train_loader, loss_function, custom_grad, truncated_length, max_grad_norm, verbose=False):
    """
    Train the model (partly) tracking the computational graph of the updates.

    Args:
        model: torch.nn.Module
        num_steps: Number of relaxation steps
        optimizer: torch.optim.Optimizer for the model.parameters()
        train_loader: DataLoader that contains the training data
        loss_function: energy.EnergyFunction giving the loss
        custom_grad: Flag whether to use model.custom_grad or resort to standard autograd
        verbose: Logging verbosity

    Returns:
        Functional version of the model (i.e. forward takes an addtional params argument)
        Parameters with the computational graph of the update steps
    """
    if truncated_length is None:
        truncated_length = num_steps

    # Convert dataloader into infinite iterator
    train_iterator = infinite_iterator(train_loader)

    # Create a functional version of the model that takes an additional `params` argument in forward
    fmodel = higher.patch.monkeypatch(model, copy_initial_weights=False, track_higher_grads=True)
    fmodel.train(mode=True)

    # First train without tracking the computational graph of updates until truncaction length
    params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]

    # Initialise the inner-level optimizer
    optimizer = create_optimizer(optimizer_name, params, {"lr": lr})

    for step in range(num_steps - truncated_length):
        # Sample training batch
        x_batch_train, y_batch_train = next(train_iterator)
        x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

        # Compute output
        output_train, state_train = fmodel(x_batch_train, params=params)
        batch_loss = loss_function(output_train, y_batch_train, fmodel, state_train)

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan.")

        optimizer.zero_grad()

        # Take inner-update step
        if custom_grad:
            grads_params = fmodel.custom_grad(
                x_batch_train, y_batch_train, state_train, loss_function, None
            )
        else:
            grads_params = torch.autograd.grad(batch_loss, params)

        for p, p_grad in zip(params, grads_params):
            p.grad = p_grad

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

        optimizer.step()

        # Logging
        if verbose and (step == 0 or (step + 1) % int(num_steps / 10) == 0):
            # HACK: Keeping track of global time by storing a variable in config.writer
            try:
                config.writer.relaxation_step += 1
            except AttributeError:
                config.writer.relaxation_step = 0
            config.writer.add_scalars('relaxation', {'batch_loss': batch_loss}, config.writer.relaxation_step)

            logging.info("step: {}/{} \t batch_loss: {:4f}".format(
                step + 1, num_steps, batch_loss))

    params = [p.detach().requires_grad_(True) for p in params]

    import copy
    model_adapted = copy.deepcopy(model)
    with torch.no_grad():
        for p, p_adapted in zip(model_adapted.parameters(), params):
            p.copy_(p_adapted)

    # Make the optimizer differentiable for the last truncated_length steps
    # NOTE: This resets the optimizer state
    optimizer_diff = make_optimizer_differentiable(
        optimizer=create_optimizer(optimizer_name, params, {"lr": lr}),
        params=params,
    )

    # Store the params_init wrt which we want higher order gradients
    params_init = list(fmodel.parameters())
    params = [p.clone() for p in params_init]

    for step in range(num_steps - truncated_length, num_steps):
        # Sample training batch
        x_batch_train, y_batch_train = next(train_iterator)
        x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

        # Compute output
        output_train, state_train = fmodel(x_batch_train, params=params)
        batch_loss = loss_function(output_train, y_batch_train, fmodel, state_train)

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan.")

        # Take inner-update step
        if custom_grad:
            grad_params = fmodel.custom_grad(x_batch_train, y_batch_train, state_train, loss_function, None)

            # HACK: Overwrite the gradients the optimizer internally computes using autograd
            #       with the manually computed ones by invoking grad_callback
            def grad_callback(internal_grads):
                del internal_grads
                return grad_params
            params = optimizer_diff.step(batch_loss, params, grad_callback=grad_callback)

        else:
            params = optimizer_diff.step(batch_loss, params)

        # Logging
        if verbose and (step == 0 or (step + 1) % int(num_steps / 10) == 0):
            # HACK: Keeping track of global time by storing a variable in config.writer
            try:
                config.writer.relaxation_step += 1
            except AttributeError:
                config.writer.relaxation_step = 0
            config.writer.add_scalars('relaxation', {'batch_loss': batch_loss}, config.writer.relaxation_step)

            logging.info("step: {}/{} \t batch_loss: {:4f}".format(
                step + 1, num_steps, batch_loss))

    return fmodel, params, params_init
