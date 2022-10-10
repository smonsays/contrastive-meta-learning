"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import copy
import json
import logging
import math
import os
import sys
import time

import torch

import data
import energy
import meta
import models
import train
import utils

from utils import config


def load_default_config(dataset, model):
    """
    Load default parameter configuration from file.

    Returns:
        Dictionary of default parameters for the given task
    """
    if dataset == "cifar10" and model == "lenet_l2":
        default_config = "config/cml_cifar10_lenet_l2.json"
    elif dataset == "mnist" and model == "mlp_l2":
        default_config = "config/cml_mnist_mlp_l2.json"
    else:
        raise ValueError(
            "Default configuration for dataset \"{}\" and model \"{}\" not defined.".format(
                dataset, model
            )
        )

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_arguments(args):
    """
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="Contrastive metalearning hyperparameters.")

    # Default model and dataset are defined here
    parser.add_argument("--dataset", choices=["mnist", "cifar10"],
                        default="mnist", help="Dataset.")

    parser.add_argument("--model", choices=["bnn", "lenet_l2", "mlp_cons", "mlp_init", "mlp_l2", "mlp-fa_l2"],
                        default="mlp_l2", help="Neural network model type.")

    # Defaults for other arguments taken from corresponding config in ./config/
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")

    parser.add_argument("--beta", type=float, default=argparse.SUPPRESS,
                        help="Nudging strength of Equilibrium Propagation.")

    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")

    parser.add_argument("--ep_variant", choices=["standard", "second_order", "symmetric"],
                        default=argparse.SUPPRESS, help="Equilibrium Propagation variant.")

    parser.add_argument("--inner_init", choices=["reset", "fixed_seed", "from_theta"],
                        default=argparse.SUPPRESS, help="Mode of initialising the inner loop.")

    parser.add_argument("--lr_inner", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the free phase.")

    parser.add_argument("--lr_nudged", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the nudged phase.")

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
                        help="Number of free relaxation steps.")

    parser.add_argument("--steps_outer", type=int, default=argparse.SUPPRESS,
                        help="Number of meta steps.")

    parser.add_argument("--steps_nudged", type=int, default=argparse.SUPPRESS,
                        help="Number of nudged relaxation steps.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def run_hyperopt(cfg, raytune=False):
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Create the training, validation and test dataloader
    train_loader, valid_loader, test_loader = data.get_dataloader(
        cfg["dataset"], batch_size=cfg["batch_size"], validation_split=cfg["validation_split"], train_subset=cfg["train_subset"]
    )

    # Create the eqprop estimator
    eqprop = meta.EquilibriumPropagation(cfg["beta"], cfg["ep_variant"])

    # Initialise the model, hyperparameters and energy functions
    if cfg["model"] == "mlp_cons":
        model = models.MultilayerPerceptron(
            cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity'])
        ).to(config.device)

        hyperparams = meta.HyperparameterDict({
            "lambda": [torch.full_like(p, math.log(cfg["init_lambda"])) for p in model.parameters()],
            "omega": [p.detach().clone() for p in model.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init="from_theta", theta_key="omega")

        energy_function = energy.CrossEntropy() + energy.ElasticRegularizer()
        cost_function = energy.CrossEntropy()

    elif cfg["model"] == "mlp_init":
        model = models.MultilayerPerceptron(
            cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity'])
        ).to(config.device)

        hyperparams = meta.HyperparameterDict({
            "omega": [p.detach().clone() for p in model.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init="from_theta", theta_key="omega")

        energy_function = energy.CrossEntropy() + energy.ElasticRegularizer(cfg["l2_strength"])
        cost_function = energy.CrossEntropy()

    elif cfg["model"] == "mlp_l2":
        model = models.MultilayerPerceptron(
            cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity'])
        ).to(config.device)

        hyperparams = meta.HyperparameterDict({
            "l2": [torch.full_like(p, cfg["init_l2"]) for p in model.parameters()]
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init=cfg["inner_init"], nonnegative_keys={"l2"})

        energy_function = energy.CrossEntropy() + energy.L2Regularizer()
        cost_function = energy.CrossEntropy()

    elif cfg["model"] == "mlp-fa_l2":
        model = models.MultilayerPerceptron(
            cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity']), feedback_alignment=True
        ).to(config.device)

        hyperparams = meta.HyperparameterDict({
            "l2": [torch.full_like(p, cfg["init_l2"]) for p in model.parameters()]
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init=cfg["inner_init"], nonnegative_keys={"l2"})

        energy_function = energy.CrossEntropy() + energy.L2Regularizer()
        cost_function = energy.CrossEntropy()

    elif cfg["model"] == "bnn":
        model = models.BayesianMultilayerPerceptron(
            cfg['dimensions'], utils.create_nonlinearity(cfg['nonlinearity']),
            cfg["init_bnn_stdv"], radial=cfg["radial"]
        ).to(config.device)

        hyperparams = meta.HyperparameterDict({
            "prior": dict([(name.replace(".", "_"), p.detach().clone()) for name, p in model.named_parameters()]),
            "prior_strength": [torch.full((len(model.layers),), cfg["init_prior_strength"])],
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init="from_theta", theta_key="prior", nonnegative_keys={"prior_strength"})

        energy_function = energy.EvidenceLowerBound(len(train_loader))
        cost_function = energy.CrossEntropy()

    elif cfg["model"] == "lenet_l2":
        model = models.LeNet(output_dim=10).to(config.device)
        hyperparams = meta.HyperparameterDict({
            "l2": [torch.full_like(p, cfg["init_l2"]) for p in model.parameters()]
        }).to(config.device)

        meta_model = meta.Hyperopt(model, hyperparams, inner_init=cfg["inner_init"], nonnegative_keys={"l2"})

        energy_function = energy.CrossEntropy() + energy.L2Regularizer()
        cost_function = energy.CrossEntropy()

    else:
        raise ValueError("Model type \"{}\" undefined".format(cfg["model"]))

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(
        cfg["optimizer_outer"], hyperparams.parameters(), {"lr": cfg["lr_outer"]}
    )

    results = {
        "grad_norm_free": torch.zeros(cfg['steps_outer'] + 1),
        "grad_norm_nudged": torch.zeros(cfg['steps_outer'] + 1),
        "param_dist": torch.zeros(cfg['steps_outer'] + 1),
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

        # Initialize the inner-level optimizer
        optimizer_inner = utils.create_optimizer(
            cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
        )

        # Initialise the inner-level scheduler if it was specified in cfg
        if "scheduler_inner" in cfg:
            scheduler_inner = utils.create_scheduler(cfg["scheduler_inner"], optimizer_inner, {
                "decay": cfg["scheduler_decay"], "step": cfg["scheduler_step"], "end": cfg["steps_inner"]
            })
        else:
            scheduler_inner = None

        # Free phase relaxation
        logging.info("Run free phase with beta=0")
        grad_norm_free = train.train_augmented(
            meta_model, cfg["steps_inner"], optimizer_inner, train_loader,
            energy_function, scheduler=scheduler_inner, verbose=not raytune
        )
        grad_hyperparams_free = train.grad_hyperparams(meta_model, train_loader, energy_function)

        if step_outer < cfg['steps_outer']:
            # Save the free-phase model state
            model_free_state = copy.deepcopy(model.state_dict())

            # Initialise the nudged phase optimizer
            optimizer_nudged = utils.create_optimizer(
                cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_nudged"]}
            )

            # Nudged phase relaxation
            logging.info("Run nudged phase with beta={:2f}".format(cfg["beta"]))
            grad_hyperparams_nudged_dict = {}
            grad_norm_nudged = train.train_augmented(
                meta_model, cfg["steps_nudged"], optimizer_nudged, train_loader, energy_function,
                cfg['beta'], valid_loader, cost_function, scheduler=None, verbose=not raytune
            )
            grad_hyperparams_nudged_dict[cfg["beta"]] = train.grad_hyperparams(
                meta_model, train_loader, energy_function, cfg['beta'], valid_loader, cost_function
            )

            if cfg['ep_variant'] == "standard":
                pass

            elif cfg['ep_variant'] == "second_order":
                # Initialise the second nudged phase optimizer (different learning rate)
                optimizer_nudged = utils.create_optimizer(
                    cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_nudged2"]}
                )

                # Run additional nudged phase with 2x nudging strength
                logging.info("Run second-order nudged phase with beta=2*{:2f}".format(cfg["beta"]))
                train.train_augmented(
                    meta_model, cfg["steps_nudged"], optimizer_nudged, train_loader, energy_function,
                    2.0 * cfg['beta'], valid_loader, cost_function, scheduler=None, verbose=not raytune
                )
                grad_hyperparams_nudged_dict[2.0 * cfg["beta"]] = train.grad_hyperparams(
                    meta_model, train_loader, energy_function, 2.0 * cfg['beta'], valid_loader, cost_function
                )

            elif cfg['ep_variant'] == "symmetric":
                # Restore the model state to free-phase fixed point
                model.load_state_dict(model_free_state)

                # Re-initialise the nudged phase optimizer (same learning rate)
                optimizer_nudged = utils.create_optimizer(
                    cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_nudged"]}
                )

                # Run additional nudged phase with negative nudging strength
                logging.info("Run symmetric nudged phase with beta=-{:2f}".format(cfg["beta"]))
                train.train_augmented(
                    meta_model, cfg["steps_nudged"], optimizer_nudged, train_loader, energy_function,
                    -1.0 * cfg['beta'], valid_loader, cost_function, scheduler=None, verbose=not raytune
                )
                grad_hyperparams_nudged_dict[-1.0 * cfg["beta"]] = train.grad_hyperparams(
                    meta_model, train_loader, energy_function, -1.0 * cfg['beta'], valid_loader, cost_function
                )

            else:
                raise ValueError("EP variant \"{}\" not defined.".format(cfg['ep_variant']))

            # Compute the gradient wrt hyperparameters
            hyperparams_grad = eqprop.hypergrad(grad_hyperparams_free, grad_hyperparams_nudged_dict)

            # Apply the outer optimisation step
            optimizer_outer.zero_grad()
            for hp, hp_grad in zip(hyperparams.parameters(), hyperparams_grad):
                hp.grad = hp_grad

            optimizer_outer.step()

            # Enforce non-negativity through projected GD for selected hyperparameters
            meta_model.enforce_nonnegativity()

            # Restore the model state to free-phase fixed point
            model_nudged_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(model_free_state)

        # Logging
        with torch.no_grad():
            test_acc = train.accuracy(model, test_loader)
            test_loss = train.loss(meta_model, test_loader, cost_function)

            train_acc = train.accuracy(model, train_loader)
            train_loss = train.loss(meta_model, train_loader, energy_function)

            valid_acc = train.accuracy(model, valid_loader)
            valid_loss = train.loss(meta_model, valid_loader, cost_function)

            param_dist_all = {
                key: torch.linalg.norm(model_nudged_state[key] - model_free_state[key])
                for key, _ in model.named_parameters()
            }
            param_dist_mean = torch.tensor(list(param_dist_all.values())).mean()

        if raytune:
            from ray import tune
            tune.report(**{
                "test_acc": test_acc.item(),
                "test_loss": test_loss.item(),
                "train_acc": train_acc.item(),
                "train_loss": train_loss.item(),
                "valid_acc": valid_acc.item(),
                "valid_loss": valid_loss.item(),
                "param_dist": param_dist_mean.item()
            })
        else:
            logging.info("step_outer: {}/{}\t train_acc: {:4f} \t valid_acc: {:4f} test_acc: {:4f}\n".format(
                step_outer, cfg['steps_outer'], train_acc, valid_acc, test_acc)
            )

            results["test_acc"][step_outer] = test_acc
            results["test_loss"][step_outer] = test_loss
            results["train_acc"][step_outer] = train_acc
            results["train_loss"][step_outer] = train_loss
            results["valid_acc"][step_outer] = valid_acc
            results["valid_loss"][step_outer] = valid_loss
            results["param_dist"][step_outer] = param_dist_mean
            results["grad_norm_free"][step_outer] = grad_norm_free
            results["grad_norm_nudged"][step_outer] = grad_norm_nudged

            config.writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc, 'valid': valid_acc}, step_outer)
            config.writer.add_scalars('loss', {'train': train_loss, 'test': test_loss, 'valid': valid_loss}, step_outer)
            config.writer.add_scalars('grad_norm', {'free': grad_norm_free, 'nudged': grad_norm_nudged}, step_outer)

            for name, p in model.named_parameters():
                config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

            for name, p in hyperparams.named_parameters():
                config.writer.add_histogram('hyperparameter/{}'.format(name), p.view(-1), step_outer)

            config.writer.add_scalars('param_dist', {
                group: torch.tensor([dist for name, dist in param_dist_all.items() if group in name]).mean()
                for group in utils.module_group_keys(model)
            }, step_outer)

    # Final validation with longer inner loop
    meta_model.reset_parameters()
    optimizer_inner = utils.create_optimizer(
        cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
    )
    if "scheduler_inner" in cfg:
        scheduler_inner = utils.create_scheduler(cfg["scheduler_inner"], optimizer_inner, {
            "decay": cfg["scheduler_decay"], "step": cfg["scheduler_step"], "end": cfg["steps_inner_long"]
        })
    else:
        scheduler_inner = None

    # Train for a longer time than during meta-training
    train.train_augmented(
        meta_model, cfg["steps_inner_long"], optimizer_inner, train_loader,
        energy_function, scheduler=scheduler_inner, verbose=not raytune
    )

    with torch.no_grad():
        test_long_acc = train.accuracy(model, test_loader)
        test_long_loss = train.loss(meta_model, test_loader, cost_function)

        train_long_acc = train.accuracy(model, train_loader)
        train_long_loss = train.loss(meta_model, train_loader, energy_function)

        valid_long_acc = train.accuracy(model, valid_loader)
        valid_long_loss = train.loss(meta_model, valid_loader, cost_function)

    # Final Testing
    logging.info("Final training on full dataset (train + valid)")

    # Concatenate the full dataset (train + valid)
    full_train_loader = data.create_multitask_loader([train_loader.dataset, valid_loader.dataset], cfg["batch_size"])

    # Initialise the model parameters
    meta_model.reset_parameters()

    # Initialize the inner-level optimizer
    optimizer_inner = utils.create_optimizer(
        cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
    )

    # Initialise the inner-level scheduler if it was specified in cfg
    if "scheduler_inner" in cfg:
        scheduler_inner = utils.create_scheduler(cfg["scheduler_inner"], optimizer_inner, {
            "decay": cfg["scheduler_decay"], "step": cfg["scheduler_step"], "end": cfg["steps_inner"]
        })
    else:
        scheduler_inner = None

    # Train on the full training dataset
    train.train_augmented(
        meta_model, cfg["steps_inner"], optimizer_inner, full_train_loader,
        energy_function, scheduler=scheduler_inner, verbose=not raytune
    )

    with torch.no_grad():
        train_full_acc = train.accuracy(model, full_train_loader)
        train_full_loss = train.loss(meta_model, full_train_loader, energy_function)

        test_full_acc = train.accuracy(model, test_loader)
        test_full_loss = train.loss(meta_model, test_loader, cost_function)

    if raytune:
        tune.report(**{
            "test_acc": test_acc.item(),
            "test_loss": test_loss.item(),
            "train_acc": train_acc.item(),
            "train_loss": train_loss.item(),
            "valid_acc": valid_acc.item(),
            "valid_loss": valid_loss.item(),
            "test_long_acc": test_long_acc.item(),
            "test_long_loss": test_long_loss.item(),
            "train_long_acc": train_long_acc.item(),
            "train_long_loss": train_long_loss.item(),
            "valid_long_acc": valid_long_acc.item(),
            "valid_long_loss": valid_long_loss.item(),
            "test_full_acc": test_full_acc.item(),
            "test_full_loss": test_full_loss.item(),
            "train_full_acc": train_full_acc.item(),
            "train_full_loss": train_full_loss.item(),
            "param_dist": param_dist_mean.item()
        })

    else:
        results["test_long_acc"] = test_long_acc
        results["test_long_loss"] = test_long_loss

        results["train_long_acc"] = train_long_acc
        results["train_long_loss"] = train_long_loss

        results["valid_long_acc"] = valid_long_acc
        results["train_long_loss"] = train_long_loss

        results["test_full_acc"] = test_full_acc
        results["test_full_loss"] = test_full_loss
        results["train_full_acc"] = train_full_acc
        results["train_full_loss"] = train_full_loss
        config.writer.add_histogram("param_dist", results["param_dist"])

        return results, model, hyperparams


if __name__ == '__main__':
    # Load configuration
    user_config = parse_arguments(sys.argv[1:])
    cfg = load_default_config(user_config["dataset"], user_config["model"])
    cfg.update(user_config)

    # Setup logging
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_hyperopt_" + cfg["dataset"] + "_" + cfg["model"]
    utils.setup_logging(run_id, cfg["log_dir"])

    # Main
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    results, model, hyperparams = run_hyperopt(cfg, raytune=False)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    results['cfg'], results['model'], results['hyperparameter'] = cfg, model.state_dict(), hyperparams.state_dict()
    torch.save(results, os.path.join(config.LOG_DIR, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.LOG_DIR, run_id + "_tensorboard")
    utils.zip_and_remove((path_tensorboard))
