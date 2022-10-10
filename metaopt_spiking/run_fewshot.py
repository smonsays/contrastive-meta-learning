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
    if dataset == "sinusoid" and model == "rsnn":
        config_name = "config/cml_sinusoid_rsnn.json"
    else:
        raise ValueError(
            "Default configuration for dataset \"{}\" and model \"{}\" not defined.".format(
                dataset, model
            )
        )

    with open(config_name) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_arguments(args):
    """
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="Contrastive metalearning hyperparameters.")

    # Default model and dataset are defined here
    parser.add_argument("--dataset", choices=["sinusoid"],
                        default="sinusoid", help="Dataset.")

    parser.add_argument("--model", choices=["mlp_cons", "mlp_init", "mlp_l2", "snn", "rsnn"],
                        default="rsnn", help="Neural network model type.")

    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")

    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def meta_test(meta_model, meta_loader, energy_function, cost_function, steps_train, batch_size, optmizer_name, optimizer_kwargs):
    meta_train_metrics = torch.zeros(len(meta_loader))
    meta_test_metrics = torch.zeros(len(meta_loader))

    for task_idx, ((x_train_batch, y_train_batch), (x_test_batch, y_test_batch)) in enumerate(meta_loader):
        x_train, y_train = x_train_batch.squeeze(0), y_train_batch.squeeze(0)
        x_test, y_test = x_test_batch.squeeze(0), y_test_batch.squeeze(0)

        # Initialise the parameters of the base learner
        meta_model.reset_parameters()

        # Initialize the inner-level optimizer
        optimizer = utils.create_optimizer(optmizer_name, meta_model.base_parameters(), optimizer_kwargs)

        # Wrap the training data into a dataloader
        train_loader = data.tensors_to_loader([x_train, y_train], shuffle=True, batch_size=batch_size)

        # Train the current task
        train.train_augmented(meta_model, steps_train, optimizer, train_loader, energy_function)

        # Validation
        with torch.no_grad():
            if meta_loader.dataset.type == "classification":
                meta_train_metrics[task_idx] = train.accuracy(meta_model, [(x_train, y_train)])
                meta_test_metrics[task_idx] = train.accuracy(meta_model, [(x_test, y_test)])

            elif meta_loader.dataset.type == "regression":
                meta_train_metrics[task_idx] = train.loss(meta_model, [(x_train, y_train)], energy_function)
                meta_test_metrics[task_idx] = train.loss(meta_model, [(x_test, y_test)], cost_function)
            else:
                raise ValueError("Dataset type \"{}\" not defined.".format(meta_loader.dataset.type))

    return meta_train_metrics, meta_test_metrics


def run_fewshot(cfg, raytune=False, checkpoint_dir=None):
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Create the training, validation and test dataloader
    meta_train_loader, meta_valid_loader, meta_test_loader = data.get_dataloader(
        cfg["dataset"], meta_batch_size=cfg["meta_batch_size"], num_batches=cfg["steps_outer"], **cfg["dataset_kwargs"]
    )
    # Create the eqprop estimator
    eqprop = meta.EquilibriumPropagation(cfg["beta"], cfg["ep_variant"])

    # Determine the neural network dimensions based on the dataset dimensions
    network_dims = [meta_train_loader.dataset.input_dim] + cfg["hidden_layers"] + [meta_train_loader.dataset.output_dim]

    # Initialise the base energy and cost function (to be augmented based on the chosen model)
    if meta_train_loader.dataset.type == "classification":
        energy_function = energy.CrossEntropy()
        cost_function = energy.CrossEntropy()

    elif meta_train_loader.dataset.type == "regression":
        energy_function = energy.MeanSquaredError()
        cost_function = energy.MeanSquaredError()

    else:
        raise ValueError("Dataset type \"{}\" not defined.".format(meta_train_loader.dataset.type))

    # Initialise the meta-model
    if cfg["model"] == "mlp_cons":
        base_learner = models.MultilayerPerceptron(network_dims, utils.create_nonlinearity(cfg['nonlinearity'])).to(config.device)
        meta_learner = meta.HyperparameterDict({
            "l2_strength": [torch.full_like(p, math.log(cfg["l2_strength"])) for p in base_learner.parameters()],
            "omega": [p.detach().clone() for p in base_learner.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(base_learner, meta_learner, inner_init="from_theta", theta_key="omega")
        energy_function += energy.ElasticRegularizer()

    elif cfg["model"] == "mlp_init":
        base_learner = models.MultilayerPerceptron(network_dims, utils.create_nonlinearity(cfg['nonlinearity'])).to(config.device)
        meta_learner = meta.HyperparameterDict({
            "omega": [p.detach().clone() for p in base_learner.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(base_learner, meta_learner, inner_init="from_theta", theta_key="omega")
        energy_function += energy.ElasticRegularizer(cfg["l2_strength"])

    elif cfg["model"] == "mlp_l2":
        base_learner = models.MultilayerPerceptron(network_dims, utils.create_nonlinearity(cfg['nonlinearity'])).to(config.device)
        meta_learner = meta.HyperparameterDict({
            "l2": [torch.full_like(p, cfg["init_l2"]) for p in base_learner.parameters()]
        }).to(config.device)

        meta_model = meta.Hyperopt(base_learner, meta_learner, inner_init=cfg["inner_init"], nonnegative_keys={"l2"})
        energy_function += energy.L2Regularizer()

    elif cfg["model"] == "snn":
        base_learner = models.SpikingNetwork(
            network_dims, {"membrane": cfg["tau_membrane"], "synapse": cfg["tau_synapse"]},
            cfg["step_size"]
        ).to(config.device)

        meta_learner = meta.HyperparameterDict({
            "omega": [p.detach().clone() for p in base_learner.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(base_learner, meta_learner, inner_init="from_theta", theta_key="omega")
        energy_function += energy.ElasticRegularizer(cfg["l2_strength"]) + energy.MeanFiringRate(target=cfg["activity_reg_target"], strength=cfg["activity_reg_strength"])

    elif cfg["model"] == "rsnn":
        base_learner = models.RecurrentSpikingNetwork(
            network_dims, cfg["tau_hidden"], cfg["tau_output"], cfg["step_size"],
            cfg["lr_modulate"], feedback_align=cfg["feedback_align"]
        ).to(config.device)

        meta_learner = meta.HyperparameterDict({
            "omega": [p.detach().clone() for p in base_learner.parameters()],
        }).to(config.device)

        meta_model = meta.Hyperopt(base_learner, meta_learner, inner_init="from_theta", theta_key="omega")
        energy_function += energy.ElasticRegularizer(cfg["l2_strength"]) + energy.MeanFiringRate(target=cfg["activity_reg_target"], strength=cfg["activity_reg_strength"])

    else:
        raise ValueError("Model type \"{}\" undefined".format(cfg["model"]))

    # Load pretrained model from checkpoint if specified
    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        meta_model.load_state_dict(torch.load(path))

    # Immediately return the initialised model if skip meta training is specified
    if "skip_meta_training" in cfg:
        return {
            "cost_function": cost_function,
            "energy_function": energy_function,
            "meta_model": meta_model,
            "meta_test_loader": meta_test_loader,
            "meta_valid_loader": meta_valid_loader,
        }

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(
        cfg["optimizer_outer"], meta_model.meta_parameters(), {"lr": cfg["lr_outer"]}
    )

    results = {
        "grad_norm_free": torch.zeros(len(meta_train_loader)),
        "grad_norm_nudged": torch.zeros(len(meta_train_loader)),
        "meta_train_train_metrics": torch.zeros(len(meta_train_loader), meta_train_loader.batch_size),
        "meta_train_test_metrics": torch.zeros(len(meta_train_loader), meta_train_loader.batch_size),
        "meta_valid_train_metrics": torch.zeros(len(meta_train_loader), len(meta_valid_loader)),
        "meta_valid_test_metrics": torch.zeros(len(meta_train_loader), len(meta_valid_loader)),
        "meta_test_train_metrics": None,
        "meta_test_test_metrics": None,
    }

    meta_valid_test_metric_best = None
    meta_model_best = copy.deepcopy(meta_model.state_dict())

    # Meta-Training
    for step_outer, ((x_train_batch, y_train_batch), (x_test_batch, y_test_batch)) in enumerate(meta_train_loader):
        hyperparams_grad_buffer = [torch.zeros_like(hp) for hp in meta_model.meta_parameters()]

        grad_norm_free = torch.zeros(len(x_train_batch))
        grad_norm_nudged = torch.zeros(len(x_train_batch))
        meta_train_train_metrics = torch.zeros(len(x_train_batch))
        meta_train_test_metrics = torch.zeros(len(x_test_batch))

        # For each task, train a few steps and compute the hypergrad
        for task, (x_train, y_train, x_test, y_test) in enumerate(zip(x_train_batch, y_train_batch, x_test_batch, y_test_batch)):

            # Initialise the model parameters
            meta_model.reset_parameters()

            # Initialize the inner-level optimizer
            optimizer_inner = utils.create_optimizer(
                cfg["optimizer_inner"], meta_model.base_parameters(), {"lr": cfg["lr_inner"]}
            )

            # Wrap the training data into a dataloader
            train_loader = data.tensors_to_loader([x_train, y_train], shuffle=True, batch_size=cfg["batch_size"])

            # Free phase relaxation
            grad_norm_free[task] = train.train_augmented(
                meta_model, cfg["steps_inner"], optimizer_inner, train_loader, energy_function
            )

            # Collect the gradients wrt hyperparameters on the free fixed-point
            grad_hyperparams_free = train.grad_hyperparams(meta_model, train_loader, energy_function)

            # Save the free-phase model state
            model_free_state = copy.deepcopy(meta_model.base_learner.state_dict())

            # Run the nudged phases
            if cfg['ep_variant'] == "standard":
                grad_hyperparams_nudged_dict = {cfg["beta"]: None}
            elif cfg['ep_variant'] == "symmetric":
                grad_hyperparams_nudged_dict = {cfg["beta"]: None, -cfg["beta"]: None}
            else:
                raise ValueError("EP variant \"{}\" not defined.".format(cfg['ep_variant']))

            # Wrap the test data into a dataloader
            test_loader = data.tensors_to_loader([x_test, y_test], shuffle=True, batch_size=cfg["batch_size"])

            for nudging_strength in grad_hyperparams_nudged_dict:
                # Restore the model state to free-phase fixed point
                meta_model.base_learner.load_state_dict(model_free_state)

                # Initialise the nudged phase optimizer
                optimizer_nudged = utils.create_optimizer(
                    cfg["optimizer_inner"], meta_model.base_parameters(), {"lr": cfg["lr_nudged"]}
                )

                # Run nudged phase with current nudging strength
                grad_norm_nudged[task] = train.train_augmented(
                    meta_model, cfg["steps_nudged"], optimizer_nudged, train_loader, energy_function,
                    nudging_strength, test_loader, cost_function
                )
                # Collect the gradients wrt hyperparameters on the nudged fixed-point
                grad_hyperparams_nudged_dict[nudging_strength] = train.grad_hyperparams(
                    meta_model, train_loader, energy_function, nudging_strength, test_loader, cost_function
                )

            # Compute the gradient wrt hyperparameters and add to buffer
            hyperparams_grad = eqprop.hypergrad(grad_hyperparams_free, grad_hyperparams_nudged_dict)
            for i, hp_grad in enumerate(hyperparams_grad):
                hyperparams_grad_buffer[i] += hp_grad

            # Restore the model state to free-phase fixed point
            meta_model.base_learner.load_state_dict(model_free_state)

            # Validation
            with torch.no_grad():
                if meta_train_loader.dataset.type == "classification":
                    meta_train_train_metrics[task] = train.accuracy(meta_model, [(x_train, y_train)])
                    meta_train_test_metrics[task] = train.accuracy(meta_model, [(x_test, y_test)])

                elif meta_train_loader.dataset.type == "regression":
                    meta_train_train_metrics[task] = train.loss(meta_model, [(x_train, y_train)], energy_function)
                    meta_train_test_metrics[task] = train.loss(meta_model, [(x_test, y_test)], cost_function)
                else:
                    raise ValueError("Dataset type \"{}\" not defined.".format(meta_train_loader.dataset.type))

        # Average hyperparam update and apply
        optimizer_outer.zero_grad()
        for hp, hp_grad in zip(meta_model.meta_parameters(), hyperparams_grad):
            hp.grad = hp_grad / len(x_train_batch)

        optimizer_outer.step()

        # Enforce non-negativity through projected GD for selected hyperparameters
        meta_model.enforce_nonnegativity()

        # Meta-Validation
        meta_valid_train_metrics, meta_valid_test_metrics = meta_test(
            meta_model, meta_valid_loader, energy_function, cost_function, cfg["steps_inner"],
            cfg["batch_size"], cfg["optimizer_inner"], optimizer_kwargs={"lr": cfg["lr_inner"]}
        )

        # Summary statsistics
        grad_norm_free_stdv, grad_norm_free_mean = torch.std_mean(grad_norm_free)
        grad_norm_nudged_stdv, grad_norm_nudged_mean = torch.std_mean(grad_norm_nudged)
        meta_train_train_metric_stdv, meta_train_train_metric_mean = torch.std_mean(meta_train_train_metrics)
        meta_train_test_metric_stdv, meta_train_test_metric_mean = torch.std_mean(meta_train_test_metrics)
        meta_valid_train_metric_stdv, meta_valid_train_metric_mean  = torch.std_mean(meta_valid_train_metrics)
        meta_valid_test_metric_stdv, meta_valid_test_metric_mean  = torch.std_mean(meta_valid_test_metrics)

        # Save best model
        if utils.is_metric_better(
            meta_valid_test_metric_mean, meta_valid_test_metric_best,
            max=True if meta_train_loader.dataset.type == "classification" else False
        ):
            meta_model_best = copy.deepcopy(meta_model.state_dict())
            meta_valid_test_metric_best = meta_valid_test_metric_mean

        # Logging
        if raytune:
            from ray import tune
            if step_outer == 0:
                meta_train_test_metric_baseline = meta_train_test_metric_mean
            meta_train_test_metric_delta = meta_train_test_metric_mean - meta_train_test_metric_baseline

            tune.report(**{
                "meta_train_train_metric": meta_train_train_metric_mean.item(),
                "meta_train_test_metric": meta_train_test_metric_mean.item(),
                "meta_train_test_metric_delta": meta_train_test_metric_delta.item(),
                "meta_valid_train_metric": meta_valid_train_metric_mean.item(),
                "meta_valid_test_metric": meta_valid_test_metric_mean.item(),
            })
            if (step_outer % (len(meta_train_loader) // 10) == 0) or (step_outer == (len(meta_train_loader) - 1)):
                with tune.checkpoint_dir(step=step_outer) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(meta_model.state_dict(), path)

        else:
            results["grad_norm_free"][step_outer] = grad_norm_free_mean
            results["grad_norm_nudged"][step_outer] = grad_norm_nudged_mean
            results["meta_train_train_metrics"][step_outer] = meta_train_train_metrics
            results["meta_train_test_metrics"][step_outer] = meta_train_test_metrics
            results["meta_valid_train_metrics"][step_outer] = meta_valid_train_metrics
            results["meta_valid_test_metrics"][step_outer] = meta_valid_test_metrics

            logging.info(
                "step_outer: {}/{}, meta_train: train: {:.4f}±{:.4f} \t test: {:.4f}±{:.4f} \t meta_valid: train: {:.4f}±{:.4f} \t test: {:.4f}±{:.4f} \t  grad_norm: free: {:.4f}±{:.4f} \t nudged: {:.4f}±{:.4f}".format(
                    step_outer, len(meta_train_loader), meta_train_train_metric_mean, meta_train_train_metric_stdv,
                    meta_train_test_metric_mean, meta_train_test_metric_stdv, meta_valid_train_metric_mean,
                    meta_valid_train_metric_stdv, meta_valid_test_metric_mean, meta_valid_test_metric_stdv,
                    grad_norm_free_mean, grad_norm_free_stdv, grad_norm_nudged_mean, grad_norm_nudged_stdv
                )
            )

            config.writer.add_scalars('meta_train', {
                'meta_train_train_metric': meta_train_train_metric_mean, 'meta_train_test_metric': meta_train_test_metric_mean
            }, step_outer)

            config.writer.add_scalars('meta_valid', {
                'meta_valid_train_metric': meta_valid_train_metric_mean, 'meta_valid_test_metric': meta_valid_test_metric_mean
            }, step_outer)

            config.writer.add_scalars('grad_norm', {'free': grad_norm_free_mean, 'nudged': grad_norm_nudged_mean}, step_outer)

            for name, p in meta_model.base_learner.named_parameters():
                config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

            for name, p in meta_model.meta_learner.named_parameters():
                config.writer.add_histogram('hyperparameter/{}'.format(name), p.view(-1), step_outer)

    # Load best model
    meta_model.load_state_dict(meta_model_best)

    # Meta-Testing
    meta_test_train_metrics, meta_test_test_metrics = meta_test(
        meta_model, meta_test_loader, energy_function, cost_function, cfg["steps_inner"],
        cfg["batch_size"], cfg["optimizer_inner"], optimizer_kwargs={"lr": cfg["lr_inner"]}
    )
    meta_valid_train_metrics, meta_valid_test_metrics = meta_test(
        meta_model, meta_valid_loader, energy_function, cost_function, cfg["steps_inner"],
        cfg["batch_size"], cfg["optimizer_inner"], optimizer_kwargs={"lr": cfg["lr_inner"]}
    )

    # Summary statistics
    meta_test_train_metric_stdv, meta_test_train_metric_mean  = torch.std_mean(meta_test_train_metrics)
    meta_test_test_metric_stdv, meta_test_test_metric_mean  = torch.std_mean(meta_test_test_metrics)
    meta_valid_train_metric_stdv, meta_valid_train_metric_mean  = torch.std_mean(meta_valid_train_metrics)
    meta_valid_test_metric_stdv, meta_valid_test_metric_mean  = torch.std_mean(meta_valid_test_metrics)

    # Logging
    if raytune:
        return {
            "meta_train_train_metric": meta_train_train_metric_mean.item(),
            "meta_train_test_metric": meta_train_test_metric_mean.item(),
            "meta_train_test_metric_delta": meta_train_test_metric_delta.item(),
            "meta_valid_train_metric": meta_valid_train_metric_mean.item(),
            "meta_valid_test_metric": meta_valid_test_metric_mean.item(),
            "meta_test_train_metric": meta_test_train_metric_mean.item(),
            "meta_test_test_metric": meta_test_test_metric_mean.item(),
        }
    else:
        results["meta_test_train_metrics"] = meta_test_train_metrics
        results["meta_test_test_metrics"] = meta_test_test_metrics

        logging.info(
            "meta_test: train: {:.4f}±{:.4f} \t test: {:.4f}±{:.4f}".format(
                meta_test_train_metric_mean, meta_test_train_metric_stdv,
                meta_test_test_metric_mean, meta_test_test_metric_stdv
            )
        )

        return results, meta_model


if __name__ == '__main__':
    # Load configuration
    user_config = parse_arguments(sys.argv[1:])
    cfg = load_default_config(user_config["dataset"], user_config["model"])
    cfg.update(user_config)

    # Setup logging
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_fewshot_" + cfg["dataset"] + "_" + cfg["model"]
    utils.setup_logging(run_id, cfg["log_dir"])

    # Main
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    results, meta_model = run_fewshot(cfg, raytune=False)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    results['cfg'], results['meta_model'] = cfg, meta_model.state_dict()
    torch.save(results, os.path.join(config.LOG_DIR, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.LOG_DIR, run_id + "_tensorboard")
    utils.zip_and_remove((path_tensorboard))
