"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
from ray import tune


import data
import energy
import meta
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
    if dataset == "mnist" and method == "cg" and model == "mlp":
        default_config = "config/cg_mnist_mlp.json"

    elif dataset == "mnist" and method == "nsa" and model == "mlp":
        default_config = "config/nsa_mnist_mlp.json"

    elif dataset == "mnist" and method == "t1t2" and model == "mlp":
        default_config = "config/t1t2_mnist_mlp.json"

    elif dataset == "cifar10" and method == "cg" and model == "lenet":
        default_config = "config/cg_cifar10_lenet.json"

    elif dataset == "cifar10" and method == "nsa" and model == "lenet":
        default_config = "config/nsa_cifar10_lenet.json"

    elif dataset == "cifar10" and method == "t1t2" and model == "lenet":
        default_config = "config/t1t2_cifar10_lenet.json"

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
                        default="mnist", help="Dataset.")

    parser.add_argument("--method", choices=["cg", "nsa", "t1t2"],
                        default="cg", help="Method to approximate the hypergradient.")

    parser.add_argument("--model", choices=["lenet", "mlp"],
                        default="mlp", help="Neural network model type.")

    # The remaining arguments set their default in the corresponding config file in config/
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")

    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")

    parser.add_argument("--inner_init", choices=["reset", "fixed_seed"],
                        default=argparse.SUPPRESS, help="Mode of initialising the inner loop.")

    parser.add_argument("--lr_inner", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the inner loop.")

    parser.add_argument("--lr_outer", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the outer loop.")

    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")

    parser.add_argument("--nonlinearity", choices=["relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")

    parser.add_argument("--optimizer_inner", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")

    parser.add_argument("--optimizer_outer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used for meta training.")

    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")

    parser.add_argument("--steps_inner", type=int, default=argparse.SUPPRESS,
                        help="Number of inner loop steps.")

    parser.add_argument("--steps_outer", type=int, default=argparse.SUPPRESS,
                        help="Number of outer loop steps.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def run_implicit(cfg, raytune=False):
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

    # Initialise the implicit gradient approximation method
    if cfg["method"] == "cg":
        implicit_gradient = meta.ConjugateGradient(cfg["cg_steps"])
    elif cfg["method"] == "nsa":
        implicit_gradient = meta.NeumannSeries(cfg["nsa_steps"], cfg["nsa_alpha"])
    elif cfg["method"] == "t1t2":
        implicit_gradient = meta.T1T2()
    else:
        raise ValueError("Implicit gradient approximation method \"{}\" undefined".format(cfg["method"]))

    # Initialise the hyperparameters
    hyperparams = meta.HyperparameterDict({
        "l2": [torch.full_like(p, cfg["init_l2"]) for p in model.parameters()]
    }).to(config.device)

    # Initialise meta model wrapping hyperparams and main model
    meta_model = meta.Hyperopt(model, hyperparams, inner_init=cfg["inner_init"], nonnegative_keys={"l2"})

    # Initialise the energy functions
    inner_loss_function = energy.CrossEntropy() + energy.L2Regularizer()
    outer_loss_function = energy.CrossEntropy()

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(
        cfg["optimizer_outer"], hyperparams.parameters(), {"lr": cfg["lr_outer"]}
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
        # Initialise the model parameters
        meta_model.reset_parameters()

        # Initialise the inner-level optimizer
        optimizer_inner = utils.create_optimizer(
            cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
        )

        # Inner-loop training
        train.train_augmented(
            meta_model, cfg["steps_inner"], optimizer_inner, train_loader, inner_loss_function, verbose=not raytune
        )

        if step_outer < cfg['steps_outer']:

            # HACK: Put all data in a single batch to save memory when backpropagating through the loss
            def dataloader_to_tensor(dataloader):
                x = torch.stack(list(zip(*list(dataloader.dataset)))[0])
                y = torch.tensor(list(zip(*list(dataloader.dataset)))[1])

                return (x, y)

            # Compute the inner and outer-loss
            inner_loss = train.loss(meta_model, [dataloader_to_tensor(train_loader)], inner_loss_function)
            outer_loss = train.loss(meta_model, [dataloader_to_tensor(valid_loader)], outer_loss_function)

            # Compute the indirect gradient wrt hyperparameters
            indirect_hypergrad = implicit_gradient.hypergrad(
                inner_loss, outer_loss, list(model.parameters()), list(hyperparams.parameters())
            )

            # Apply the outer optimisation step
            # NOTE: We assume that direct_hyper_grad == 0, i.e. the outer-loss does not depend on the hyperparams
            optimizer_outer.zero_grad()
            for hp, hp_grad in zip(hyperparams.parameters(), indirect_hypergrad):
                hp.grad = hp_grad

            optimizer_outer.step()

            # Enforce non-negativity through projected GD for selected hyperparameters
            meta_model.enforce_nonnegativity()

        # Testing
        with torch.no_grad():
            test_acc = train.accuracy(model, test_loader)
            test_loss = train.loss(model, test_loader, outer_loss_function)

            train_acc = train.accuracy(model, train_loader)
            train_loss = train.loss(model, train_loader, outer_loss_function)

            valid_acc = train.accuracy(model, valid_loader)
            valid_loss = train.loss(model, valid_loader, outer_loss_function)

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
            results["train_acc"][step_outer] = train_acc
            results["train_loss"][step_outer] = train_loss
            results["valid_acc"][step_outer] = valid_acc
            results["valid_loss"][step_outer] = valid_loss

            config.writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc, 'valid': valid_acc}, step_outer)
            config.writer.add_scalars('loss', {'train': train_loss, 'test': test_loss, 'valid': valid_loss}, step_outer)

            for name, p in model.named_parameters():
                config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

            for name, p in hyperparams.named_parameters():
                config.writer.add_histogram('hyperparameter/{}'.format(name), p.view(-1), step_outer)

    # Final Testing
    logging.info("Final training on full dataset (train + valid)")

    # Concatenate the full dataset (train + valid)
    full_train_loader = data.create_multitask_loader([train_loader.dataset, valid_loader.dataset], cfg["batch_size"])

    # Initialise the model parameters
    meta_model.reset_parameters()

    # Initialise the inner-level optimizer
    optimizer_inner = utils.create_optimizer(
        cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
    )

    # Inner-loop training
    train.train_augmented(
        meta_model, cfg["steps_inner"], optimizer_inner, full_train_loader, inner_loss_function, verbose=not raytune
    )

    # Final testing
    with torch.no_grad():
        train_acc_full = train.accuracy(meta_model, full_train_loader)
        train_loss_full = train.loss(meta_model, full_train_loader, outer_loss_function)

        test_acc_full = train.accuracy(meta_model, test_loader)
        test_loss_full = train.loss(meta_model, test_loader, outer_loss_function)

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

        return results, model, hyperparams


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
    results, model, hyperparams = run_implicit(cfg, raytune=False)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    results['cfg'], results['model'], results['hyperparameter'] = cfg, model.state_dict(), hyperparams.state_dict()
    torch.save(results, os.path.join(config.LOG_DIR, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.LOG_DIR, run_id + "_tensorboard")
    utils.zip_and_remove((path_tensorboard))
