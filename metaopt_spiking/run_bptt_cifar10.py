"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import functools
import json
import logging
import os
import sys
import time

import higher
import torch
from ray import tune


import data
import energy
import models
import train
import utils

from utils import config


def load_default_config(dataset, method, model):
    """
    Load default parameter configuration from file.

    Returns:
        Dictionary of default parameters for the given task
    """
    if dataset == "cifar10" and method == "tbptt" and model == "lenet":
        default_config = "config/tbptt_cifar10_lenet.json"

    else:
        raise ValueError("Default configuration for dataset \"{}\", method \"{}\" and model \"{}\" not defined.".format(dataset, method, model))

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_arguments(args):
    """
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="IFT-based hyperparameter optimisation.")

    # The following arguments set their default here
    parser.add_argument("--dataset", choices=["cifar10", "mnist"],
                        default="cifar10", help="Dataset.")

    parser.add_argument("--method", choices=["tbptt"],
                        default="tbptt", help="Method to approximate the hypergradient.")

    parser.add_argument("--model", choices=["lenet", "mlp"],
                        default="lenet", help="Neural network model type.")

    # The remaining arguments set their default in the corresponding config file in config/
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")

    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")

    parser.add_argument("--lr_inner", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the inner loop.")

    parser.add_argument("--lr_outer", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the outer loop.")

    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")

    parser.add_argument("--nonlinearity", choices=["relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")

    parser.add_argument("--optimizer_inner", choices=["adam", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")

    parser.add_argument("--optimizer_outer", choices=["adam", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used for meta training.")

    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")

    parser.add_argument("--steps_inner", type=int, default=argparse.SUPPRESS,
                        help="Number of inner loop steps.")

    parser.add_argument("--steps_outer", type=int, default=argparse.SUPPRESS,
                        help="Number of outer loop steps.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


class PartialFmodule:
    """
    Monkeypatch the forward method of a fmodule to use a specific set of parameters.
    """
    def __init__(self, fmodel, params):
        self.fmodel = fmodel
        self.params = params
        self.forward = functools.partial(self.fmodel.forward, params=self.params)

    def __call__(self, args):
        return self.forward(args)

    def __getattr__(self, attr):
        """
        Overwrite forward/__call__ via `getattr` hacking.
        """
        return getattr(self.fmodel, attr)


def make_optimizer_differentiable(optimizer, params):
    if isinstance(optimizer, torch.optim.Adam):
        optimizer_diff = higher.optim.DifferentiableAdam(optimizer, params)

    elif isinstance(optimizer, torch.optim.Adamax):
        optimizer_diff = higher.optim.DifferentiableAdamax(optimizer, params)

    elif isinstance(optimizer, torch.optim.SGD):
        optimizer_diff = higher.optim.DifferentiableSGD(optimizer, params)

    else:
        raise ValueError("Optimizer \"{}\" undefined".format(optimizer))

    return optimizer_diff


def train_differentiable(model, l2_strengths, num_steps, optimizer_name, lr, train_loader, truncated_length, verbose=False):
    """
    Train the model tracking the computational graph of the updates.

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
    # Convert dataloader into infinite iterator
    train_iterator = train.utils.infinite_iterator(train_loader)

    # Create a functional version of the model that takes an additional `params` argument in forward
    fmodel = higher.patch.monkeypatch(model, copy_initial_weights=False, track_higher_grads=True)
    fmodel.train(mode=True)

    # First train without tracking the computational graph of updates until truncaction length
    params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]

    # Initialise the inner-level optimizer
    optimizer = utils.create_optimizer(optimizer_name, params, {"lr": lr})

    for step in range(num_steps - truncated_length):
        # Sample training batch
        x_batch_train, y_batch_train = next(train_iterator)
        x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

        # Compute output
        output_train, state_train = fmodel(x_batch_train, params=params)
        batch_loss = torch.nn.functional.cross_entropy(output_train, y_batch_train, reduction="mean")
        batch_loss += torch.sum(torch.cat([
            (l2 * p**2).view(-1)
            for p, l2 in zip(params, l2_strengths)
        ]))

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan.")

        optimizer.zero_grad()
        grads_params = torch.autograd.grad(batch_loss, params)

        for p, p_grad in zip(params, grads_params):
            p.grad = p_grad

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

    import copy
    model_adapted = copy.deepcopy(model)
    with torch.no_grad():
        for p, p_adapted in zip(model_adapted.parameters(), params):
            p.copy_(p_adapted)

    # Make the optimizer differentiable for the last truncated_length steps
    # NOTE: This resets the optimizer state
    optimizer_diff = make_optimizer_differentiable(
        optimizer=utils.create_optimizer(optimizer_name, params, {"lr": lr}),
        params=params,
    )

    # Store the params_init wrt which we want higher order gradients
    params_init = params
    params = [p.clone() for p in params_init]

    for step in range(num_steps - truncated_length, num_steps):
        # Sample training batch
        x_batch_train, y_batch_train = next(train_iterator)
        x_batch_train, y_batch_train = x_batch_train.to(config.device), y_batch_train.to(config.device)

        # Compute output
        output_train, state_train = fmodel(x_batch_train, params=params)
        batch_loss = torch.nn.functional.cross_entropy(output_train, y_batch_train, reduction="mean")
        batch_loss += torch.sum(torch.cat([
            (l2 * p**2).view(-1)
            for p, l2 in zip(params, l2_strengths)
        ]))

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan.")

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


def run_bptt_cifar10(cfg, raytune=False):
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Create the training, validation and test dataloader
    train_loader, valid_loader, test_loader = data.get_dataloader(
        cfg["dataset"], batch_size=cfg["batch_size"], validation_split=cfg["validation_split"], train_subset=cfg["train_subset"]
    )

    # Initialise the model
    if cfg["model"] == "mlp":
        model = models.MultilayerPerceptron(cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity'])).to(config.device)
    elif cfg["model"] == "lenet":
        # NOTE: Hard-coded output_dim as all datasets considered so far have 10 outputs
        model = models.LeNet(output_dim=10).to(config.device)
    else:
        raise ValueError("Model type \"{}\" undefined".format(cfg["model"]))

    # Initialise the hyperparameters
    l2_strengths = [
        torch.full_like(p, cfg["init_l2"], requires_grad=True) for p in model.parameters()
    ]

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(
        cfg["optimizer_outer"], list(model.parameters()) + l2_strengths, {"lr": cfg["lr_outer"]}
    )

    results = {
        "test_acc": torch.zeros(cfg['steps_outer'] + 1),
        "test_loss": torch.zeros(cfg['steps_outer'] + 1),
        "train_acc": torch.zeros(cfg['steps_outer'] + 1),
        "train_loss": torch.zeros(cfg['steps_outer'] + 1),
        "valid_acc": torch.zeros(cfg['steps_outer'] + 1),
        "valid_loss": torch.zeros(cfg['steps_outer'] + 1),
    }

    # NOTE: There is one additional meta iteration which does not invoke a meta update:
    # +1 to evaluate the validation accuracy after the last outer step
    for step_outer in range(cfg['steps_outer'] + 1):

        # Inner-loop training
        model_func, params, params_init = train_differentiable(
            model=model,
            l2_strengths=l2_strengths,
            num_steps=cfg["steps_inner"],
            optimizer_name=cfg["optimizer_inner"],
            lr=cfg["lr_inner"],
            train_loader=train_loader,
            truncated_length=cfg["truncated_length"],
            verbose=not raytune
        )

        # Glue the trained parameters to the functional model for evaluation
        model_adapted = PartialFmodule(model_func, params)

        if step_outer < cfg['steps_outer']:

            # HACK: Put all data in a single batch to save memory when backpropagating through the loss
            def dataloader_to_tensor(dataloader):
                x = torch.stack(list(zip(*list(dataloader.dataset)))[0])
                y = torch.tensor(list(zip(*list(dataloader.dataset)))[1])

                return (x, y)

            # Compute outer-loss given fine-tuned model
            loss_outer = train.loss(model_adapted, [dataloader_to_tensor(test_loader)], energy.CrossEntropy())

            # Backpropagate through the subsidary learning process for the outer-loss
            params_init_grad = torch.autograd.grad(loss_outer, params_init, retain_graph=True)
            l2_strengths_grad = torch.autograd.grad(loss_outer, l2_strengths)

            # Apply the outer optimisation step
            optimizer_outer.zero_grad()

            for p_init, p_init_grad in zip(model.parameters(), params_init_grad):
                p_init.grad = p_init_grad

            for l2, l2_grad in zip(l2_strengths, l2_strengths_grad):
                l2.grad = l2_grad

            optimizer_outer.step()

            # Enforce non-negativity through projected GD for selected hyperparameters
            with torch.no_grad():
                for l2 in l2_strengths:
                    l2.relu_()

        # Testing
        with torch.no_grad():
            test_acc = train.accuracy(model_adapted, test_loader)
            test_loss = train.loss(model_adapted, test_loader, energy.CrossEntropy())

            train_acc = train.accuracy(model_adapted, train_loader)
            train_loss = train.loss(model_adapted, train_loader, energy.CrossEntropy())

            valid_acc = train.accuracy(model_adapted, valid_loader)
            valid_loss = train.loss(model_adapted, valid_loader, energy.CrossEntropy())

        # Logging
        if raytune:
            tune.report(**{
                "test_acc": test_acc.item(),
                "test_loss": test_loss.item(),
                "train_acc": train_acc.item(),
                "train_loss": train_loss.item(),
                "valid_acc": valid_acc.item(),
                "valid_loss": valid_loss.item(),
            })
        else:
            logging.info(
                "step_outer: {}/{}\t train_acc: {:4f} \t valid_acc: {:4f} \t test_acc: {:4f}".format(
                    step_outer, cfg['steps_outer'], train_acc, valid_acc, test_acc
                )
            )
            logging.info(
                "step_outer: {}/{}\t train_loss: {:4f} \t valid_loss: {:4f} \t test_loss: {:4f}".format(
                    step_outer, cfg['steps_outer'], train_loss, valid_loss, test_loss
                )
            )

            results["test_acc"][step_outer] = test_acc
            results["test_loss"][step_outer] = test_loss
            results["valid_acc"][step_outer] = valid_acc
            results["valid_loss"][step_outer] = valid_loss
            results["train_acc"][step_outer] = train_acc
            results["train_loss"][step_outer] = train_loss

            config.writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc, 'valid': valid_acc}, step_outer)
            config.writer.add_scalars('loss', {'train': train_loss, 'test': test_loss, 'valid': valid_loss}, step_outer)

            for name, p in model.named_parameters():
                config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

            for name, p in enumerate(l2_strengths):
                config.writer.add_histogram('hyperparameter/{}'.format(name), p.view(-1), step_outer)

    # Final Testing
    logging.info("Final training on full dataset (train + valid)")

    # Concatenate the full dataset (train + valid)
    full_train_loader = data.create_multitask_loader(
        [train_loader.dataset, valid_loader.dataset], cfg["batch_size"]
    )

    # Inner-loop training
    model_func, params, params_init = train_differentiable(
        model=model,
        l2_strengths=l2_strengths,
        num_steps=cfg["steps_inner"],
        optimizer_name=cfg["optimizer_inner"],
        lr=cfg["lr_inner"],
        train_loader=full_train_loader,
        truncated_length=0,
        verbose=not raytune
    )

    # Glue the trained parameters to the functional model for evaluation
    model_adapted = PartialFmodule(model_func, params)

    # Final testing
    with torch.no_grad():
        train_acc_full = train.accuracy(model_adapted, full_train_loader)
        train_loss_full = train.loss(model_adapted, full_train_loader, energy.CrossEntropy())

        test_acc_full = train.accuracy(model_adapted, test_loader)
        test_loss_full = train.loss(model_adapted, test_loader, energy.CrossEntropy())

    if raytune:
        tune.report(**{
            "test_acc": test_acc.item(),
            "test_loss": test_loss.item(),
            "train_acc": train_acc.item(),
            "train_loss": train_loss.item(),
            "valid_acc": valid_acc.item(),
            "valid_loss": valid_loss.item(),
            "test_acc_full": test_acc_full.item(),
            "test_loss_full": test_loss_full.item(),
            "train_acc_full": train_acc_full.item(),
            "train_loss_full": train_loss_full.item(),
        })

    else:
        results["test_acc_full"] = test_acc_full
        results["test_loss_full"] = test_loss_full
        results["train_acc_full"] = train_acc_full
        results["train_loss_full"] = train_loss_full

        logging.info("train_acc_full: {:4f} \t test_acc_full: {:4f}".format(train_acc_full, test_acc_full))
        logging.info("train_loss_full: {:4f} \t test_loss_full: {:4f}".format(train_loss_full, test_loss_full))

        return results, model, l2_strengths


if __name__ == '__main__':
    # Load configuration (Priority: 1 User, 2 Random, 3 Default)
    user_config = parse_arguments(sys.argv[1:])
    cfg = load_default_config(user_config["dataset"], user_config["method"], user_config["model"])
    cfg.update(user_config)

    # Setup logging
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + cfg["method"] + "_" + cfg["dataset"] + "_" + cfg["model"]
    utils.setup_logging(run_id, cfg["log_dir"])

    # Main
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    results, model, l2_strengths = run_bptt_cifar10(cfg, raytune=False)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    results['cfg'], results['model'], results['hyperparameter'] = cfg, model.state_dict(), l2_strengths
    torch.save(results, os.path.join(config.LOG_DIR, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.LOG_DIR, run_id + "_tensorboard")
    utils.zip_and_remove((path_tensorboard))
