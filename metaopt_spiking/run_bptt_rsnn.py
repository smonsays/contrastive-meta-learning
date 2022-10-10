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
import functools
import logging
import os
import sys
import time

import torch

import data
import energy
import models
import train
import utils

from utils import config


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


def load_default_config(dataset, method_outer, method_inner):
    """
    Load default parameter configuration from file.

    Returns:
        Dictionary of default parameters for the given task
    """
    if dataset == "sinusoid" and method_outer == "bptt" and method_inner == "bptt":
        default_config = "config/bptt_sinusoid_bptt.json"
    elif dataset == "sinusoid" and method_outer == "bptt" and method_inner == "eprop":
        default_config = "config/bptt_sinusoid_eprop.json"
    elif dataset == "sinusoid" and method_outer == "tbptt" and method_inner == "eprop":
        default_config = "config/tbptt_sinusoid_eprop.json"
    else:
        raise ValueError(
            "Default configuration for dataset \"{}\", method_outer \"{}\" and method_inner \"{}\" not defined.".format(
                dataset, method_outer, method_inner
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
    parser.add_argument("--dataset", choices=["sinusoid"],
                        default="sinusoid", help="Dataset.")

    parser.add_argument("--method_outer", choices=["bptt", "tbptt"],
                        default="tbptt", help="Outer learning algorithm.")

    parser.add_argument("--method_inner", choices=["bptt", "eprop"],
                        default="eprop", help="Inner learning algorithm.")

    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")

    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def meta_test(model, meta_loader, loss_function_inner, loss_function_outer, steps_train, batch_size, optmizer_name, optimizer_kwargs):

    # Save the model to be restored after testing
    model_state = copy.deepcopy(model.state_dict())

    meta_train_metrics = torch.zeros(len(meta_loader))
    meta_test_metrics = torch.zeros(len(meta_loader))

    for task_idx, ((x_train_batch, y_train_batch), (x_test_batch, y_test_batch)) in enumerate(meta_loader):
        x_train, y_train = x_train_batch.squeeze(0), y_train_batch.squeeze(0)
        x_test, y_test = x_test_batch.squeeze(0), y_test_batch.squeeze(0)

        # Restore the model state
        model.load_state_dict(model_state)

        # Initialize the inner-level optimizer
        optimizer = utils.create_optimizer(optmizer_name, model.parameters(), optimizer_kwargs)

        # Wrap the training data into a dataloader
        train_loader = data.tensors_to_loader([x_train, y_train], shuffle=True, batch_size=batch_size)

        # Train the current task
        train.train(model, steps_train, optimizer, train_loader, loss_function_inner)

        # Validation
        with torch.no_grad():
            if meta_loader.dataset.type == "classification":
                meta_train_metrics[task_idx] = train.accuracy(model, [(x_train, y_train)])
                meta_test_metrics[task_idx] = train.accuracy(model, [(x_test, y_test)])

            elif meta_loader.dataset.type == "regression":
                meta_train_metrics[task_idx] = train.loss(model, [(x_train, y_train)], loss_function_inner)
                meta_test_metrics[task_idx] = train.loss(model, [(x_test, y_test)], loss_function_outer)
            else:
                raise ValueError("Dataset type \"{}\" not defined.".format(meta_loader.dataset.type))

    # Restore the model state
    model.load_state_dict(model_state)

    return meta_train_metrics, meta_test_metrics


def run_bptt_rsnn(cfg, raytune=False):
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Create the training, validation and test dataloader
    meta_train_loader, meta_valid_loader, meta_test_loader = data.get_dataloader(
        cfg["dataset"], meta_batch_size=cfg["meta_batch_size"], num_batches=cfg["steps_outer"], **cfg["dataset_kwargs"]
    )

    # Determine the neural network dimensions based on the dataset dimensions
    network_dims = [meta_train_loader.dataset.input_dim] + cfg["hidden_layers"] + [meta_train_loader.dataset.output_dim]

    # Initialise the base energy and cost function (to be augmented based on the chosen model)
    if meta_train_loader.dataset.type == "classification":
        loss_function_inner = energy.CrossEntropy()
        loss_function_outer = energy.CrossEntropy()

    elif meta_train_loader.dataset.type == "regression":
        loss_function_inner = energy.MeanSquaredError()
        loss_function_outer = energy.MeanSquaredError()

    else:
        raise ValueError("Dataset type \"{}\" not defined.".format(meta_train_loader.dataset.type))

    if cfg["model"] == "rsnn":
        base_learner = models.RecurrentSpikingNetwork(
            network_dims, cfg["tau_hidden"], cfg["tau_output"], cfg["step_size"],
            cfg["lr_modulate"], feedback_align=cfg["feedback_align"]
        ).to(config.device)

        loss_function_inner += energy.MeanFiringRate(target=cfg["activity_reg_target"], strength=cfg["activity_reg_strength"])

    else:
        raise ValueError("Model type \"{}\" undefined".format(cfg["model"]))

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(
        cfg["optimizer_outer"], base_learner.parameters(), {"lr": cfg["lr_outer"]}
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

    # Meta-Training
    for step_outer, ((x_train_batch, y_train_batch), (x_test_batch, y_test_batch)) in enumerate(meta_train_loader):
        hyperparams_grad_buffer = [torch.zeros_like(hp) for hp in base_learner.parameters()]

        meta_train_train_metrics = torch.zeros(len(x_train_batch))
        meta_train_test_metrics = torch.zeros(len(x_test_batch))

        optimizer_outer.zero_grad()

        # For each task, train a few steps and compute the hypergrad
        for task, (x_train, y_train, x_test, y_test) in enumerate(zip(x_train_batch, y_train_batch, x_test_batch, y_test_batch)):

            # Wrap the training amd test data into a dataloader
            train_loader = data.tensors_to_loader([x_train, y_train], shuffle=True, batch_size=cfg["batch_size"])
            test_loader = data.tensors_to_loader([x_test, y_test], shuffle=True, batch_size=cfg["batch_size"])

            # Train the model tracking the computational graph of the updates
            base_learner_func, params, params_init = train.train_differentiable(
                model=base_learner,
                num_steps=cfg["steps_inner"],
                optimizer_name=cfg["optimizer_inner"],
                lr=cfg["lr_inner"],
                train_loader=train_loader,
                loss_function=loss_function_inner,
                custom_grad=cfg["grad_eprop"],
                truncated_length=cfg["truncated_length"],
                max_grad_norm=cfg["max_grad_norm"],
                verbose=not raytune,
            )

            # Glue the trained parameters to the functional model for evaluation
            base_learner_adapted = PartialFmodule(base_learner_func, params)

            # Compute outer-loss given fine-tuned model
            loss_outer = train.loss(base_learner_adapted, test_loader, loss_function_outer)

            # Backpropagate through the subsidary learning process for the outer-loss
            hyperparams_grad = torch.autograd.grad(loss_outer, params_init)

            for i, hp_grad in enumerate(hyperparams_grad):
                hyperparams_grad_buffer[i] += hp_grad

            # Validation
            with torch.no_grad():
                if meta_train_loader.dataset.type == "classification":
                    meta_train_train_metrics[task] = train.accuracy(base_learner_adapted, [(x_train, y_train)])
                    meta_train_test_metrics[task] = train.accuracy(base_learner_adapted, [(x_test, y_test)])

                elif meta_train_loader.dataset.type == "regression":
                    meta_train_train_metrics[task] = train.loss(base_learner_adapted, [(x_train, y_train)], loss_function_inner)
                    meta_train_test_metrics[task] = train.loss(base_learner_adapted, [(x_test, y_test)], loss_function_outer)
                else:
                    raise ValueError("Dataset type \"{}\" not defined.".format(meta_train_loader.dataset.type))

        # Average hyperparam update and apply
        optimizer_outer.zero_grad()
        for hp, hp_grad in zip(base_learner.parameters(), hyperparams_grad):
            hp.grad = hp_grad / len(x_train_batch)

        optimizer_outer.step()

        # Meta-Validation
        meta_valid_train_metrics, meta_valid_test_metrics = meta_test(
            base_learner, meta_valid_loader, loss_function_inner, loss_function_outer, cfg["steps_inner"],
            cfg["batch_size"], cfg["optimizer_inner"], optimizer_kwargs={"lr": cfg["lr_inner"]}
        )

        # Summary statsistics
        meta_train_train_metric_stdv, meta_train_train_metric_mean = torch.std_mean(meta_train_train_metrics)
        meta_train_test_metric_stdv, meta_train_test_metric_mean = torch.std_mean(meta_train_test_metrics)
        meta_valid_train_metric_stdv, meta_valid_train_metric_mean  = torch.std_mean(meta_valid_train_metrics)
        meta_valid_test_metric_stdv, meta_valid_test_metric_mean  = torch.std_mean(meta_valid_test_metrics)

        # Save best model
        if utils.is_metric_better(
            meta_valid_test_metric_mean, meta_valid_test_metric_best,
            max=True if meta_train_loader.dataset.type == "classification" else False
        ):
            best_model = copy.deepcopy(base_learner.state_dict())
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
        else:
            results["meta_train_train_metrics"][step_outer] = meta_train_train_metrics
            results["meta_train_test_metrics"][step_outer] = meta_train_test_metrics
            results["meta_valid_train_metrics"][step_outer] = meta_valid_train_metrics
            results["meta_valid_test_metrics"][step_outer] = meta_valid_test_metrics

            logging.info(
                "step_outer: {}/{}, meta_train: train: {:.4f}±{:.4f} \t test: {:.4f}±{:.4f} \t meta_valid: train: {:.4f}±{:.4f} \t test: {:.4f}±{:.4f}".format(
                    step_outer, len(meta_train_loader), meta_train_train_metric_mean, meta_train_train_metric_stdv,
                    meta_train_test_metric_mean, meta_train_test_metric_stdv, meta_valid_train_metric_mean,
                    meta_valid_train_metric_stdv, meta_valid_test_metric_mean, meta_valid_test_metric_stdv,
                )
            )

            config.writer.add_scalars('meta_train', {
                'meta_train_train_metric': meta_train_train_metric_mean, 'meta_train_test_metric': meta_train_test_metric_mean
            }, step_outer)

            config.writer.add_scalars('meta_valid', {
                'meta_valid_train_metric': meta_valid_train_metric_mean, 'meta_valid_test_metric': meta_valid_test_metric_mean
            }, step_outer)

            for name, p in base_learner.named_parameters():
                config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

    # Load best model
    base_learner.load_state_dict(best_model)

    # Meta-Testing
    meta_test_train_metrics, meta_test_test_metrics = meta_test(
        base_learner, meta_test_loader, loss_function_inner, loss_function_outer, cfg["steps_inner"],
        cfg["batch_size"], cfg["optimizer_inner"], optimizer_kwargs={"lr": cfg["lr_inner"]}
    )
    meta_valid_train_metrics, meta_valid_test_metrics = meta_test(
        base_learner, meta_valid_loader, loss_function_inner, loss_function_outer, cfg["steps_inner"],
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

    return results, base_learner


if __name__ == '__main__':
    # Load configuration
    user_config = parse_arguments(sys.argv[1:])
    cfg = load_default_config(user_config["dataset"], user_config["method_outer"], user_config["method_inner"],)
    cfg.update(user_config)

    # Setup logging
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + cfg["method_outer"] + "_rsnn_" + cfg["dataset"] + "_" + cfg["method_inner"]
    utils.setup_logging(run_id, cfg["log_dir"])

    # Main
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    results, model = run_bptt_rsnn(cfg, raytune=False)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    results['cfg'], results['model'] = cfg, model.state_dict()
    torch.save(results, os.path.join(config.LOG_DIR, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.LOG_DIR, run_id + "_tensorboard")
    utils.zip_and_remove((path_tensorboard))
