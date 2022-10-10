
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
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
from ray import tune

import data
from run_bandit import run_bandit
from run_fewshot import run_fewshot


def parse_arguments(args):
    """
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="Contrastive metalearning hyperparameters.")

    # Default model and dataset are defined here
    parser.add_argument("--method", choices=["cml", "maml"],
                        default="cml", help="Meta-learning method.")

    parser.add_argument("--meta_model", choices=["gain_mod", "imaml", "learned_init"],
                        default="imaml", help="Meta model.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def load_default_config(method, meta_model):
    """
    Load default parameter configuration.

    Returns:
        Dictionary of default parameters for the given task
    """
    if method == "cml" and meta_model == "imaml":
        return {
            "agent": "NeuralGreedy",
            "batch_size": 512,
            "beta": 0.3,
            "buffer_size": 80000,
            "context_dim": 2,
            "dataset": {
                "name": "wheel",
                "num_tasks_test": 10,
                "num_tasks_train": 64,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 100,
                "shots_test": 50,
                "shots_train": 512
            },
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "epsilon": 0.0,
            "hidden_dims": [100, 100],
            "initial_pulls": 2,
            "l2_reg": 1000.0,
            "lr_inner": 0.0001,
            "lr_nudged": 0.03,
            "lr_online": 0.0001,
            "lr_outer": 0.03,
            "meta_batch_size": 8,
            "meta_model": "imaml",
            "method": "cml",
            "num_actions": 5,
            "num_trials": 80000,
            "optim_inner": "adam",
            "optim_online": "adam",
            "optim_outer": "adam",
            "repetitions": 5,
            "reset_optim": False,
            "reset_params": False,
            "seed": tune.grid_search(list(range(2022, 2072))),
            "steps_inner": 250,
            "steps_nudged": 100,
            "steps_online": 250,
            "train_freq": 50
        }
    elif method == "cml" and meta_model == "gain_mod":
        return {
            "agent": "NeuralGreedy",
            "batch_size": 512,
            "beta": 10.0,
            "buffer_size": 80000,
            "context_dim": 2,
            "dataset": {
                "name": "wheel",
                "num_tasks_test": 10,
                "num_tasks_train": 64,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 100,
                "shots_test": 50,
                "shots_train": 512
            },
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "epsilon": 0.0,
            "hidden_dims": [100, 100],
            "initial_pulls": 2,
            "lr_inner": 0.0003,
            "lr_nudged": 0.0001,
            "lr_online": 0.0001,
            "lr_outer": 0.03,
            "meta_batch_size": 16,
            "meta_model": "gain_mod",
            "method": "cml",
            "num_actions": 5,
            "num_trials": 80000,
            "optim_inner": "sgd",
            "optim_online": "sgd_nesterov",
            "optim_outer": "adam",
            "reset_optim": False,
            "reset_params": False,
            "seed": tune.grid_search(list(range(2022, 2072))),
            "steps_inner": 1000,
            "steps_nudged": 100,
            "steps_online": 500,
            "train_freq": 100
        }

    elif method == "maml" and meta_model == "learned_init":
        return {
            "agent": "NeuralGreedy",
            "batch_size": 512,
            "buffer_size": 80000,
            "context_dim": 2,
            "dataset": {
                "name": "wheel",
                "num_tasks_test": 10,
                "num_tasks_train": 64,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 100,
                "shots_test": 50,
                "shots_train": 512
            },
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "epsilon": 0.0,
            "hidden_dims": [100, 100],
            "initial_pulls": 2,
            "l2_reg": 0.0,
            "lr_inner": 0.01,
            "lr_online": 0.001,
            "lr_outer": 0.1,
            "meta_batch_size": 32,
            "meta_model": "learned_init",
            "method": "maml",
            "num_actions": 5,
            "num_trials": 80000,
            "optim_inner": "sgd",
            "optim_online": "sgd",
            "optim_outer": "adam",
            "reset_optim": False,
            "reset_params": False,
            "seed": tune.grid_search(list(range(2022, 2072))),
            "steps_inner": 10,
            "steps_online": 100,
            "train_freq": 100
        }
    else:
        raise ValueError(
            "Default configuration for method \"{}\" and meta-model \"{}\" not defined.".format(
                method, meta_model
            )
        )


class MetaBandit(tune.Trainable):
    def setup(self, config):
        self.config_bandit = {
            "agent": config.get("agent", None),
            "a0": config.get("a0", None),
            "b0": config.get("b0", None),
            "batch_size": config.get("batch_size", None),
            "buffer_size": config.get("buffer_size", None),
            "context_dim": config.get("context_dim", None),
            "deltas": config.get("deltas", None),
            "epsilon": config.get("epsilon", None),
            "hidden_dims": config.get("hidden_dims", None),
            "initial_pulls": config.get("initial_pulls", None),
            "l2_reg": config.get("l2_reg", None),
            "lambda_prior": config.get("lambda_prior", None),
            "lr": config.get("lr_online", None),
            "lr_decay_rate": config.get("lr_online_decay_rate", None),
            "max_grad_norm": config.get("max_grad_norm", None),
            "model": config.get("meta_model", None),
            "num_actions": config.get("num_actions", None),
            "num_trials": config.get("num_trials", None),
            "num_updates": config.get("steps_online", None),
            "optimizer": config.get("optim_online", None),
            "repetitions": config.get("repetitions", None),
            "reset_optim": config.get("reset_optim", None),
            "reset_params": config.get("reset_params", None),
            "seed": config.get("seed", None),
            "train_freq": config.get("train_freq", None),
        }
        self.config_fewshot = {
            "batch_size": config.get("batch_size", None),
            "beta": config.get("beta", None),
            "dataset": config.get("dataset", None),
            "hidden_dims": config.get("hidden_dims", None),
            "l2_reg": config.get("l2_reg", None),
            "lr_inner": config.get("lr_inner", None),
            "lr_nudged": config.get("lr_nudged", None),
            "lr_outer": config.get("lr_outer", None),
            "meta_batch_size": config.get("meta_batch_size", None),
            "meta_model": config.get("meta_model", None),
            "method": config.get("method", None),
            "optim_inner": config.get("optim_inner", None),
            "optim_outer": config.get("optim_outer", None),
            "seed": config.get("seed", None),
            "steps_inner": config.get("steps_inner", None),
            "steps_nudged": config.get("steps_nudged", None),
        }

    def step(self):
        # Fewshot
        hparams, metrics_fewshot = run_fewshot(self.config_fewshot)
        valid_loss_outer_mean = jnp.mean(metrics_fewshot["valid_loss_outer"][-1])
        test_loss_outer_mean = jnp.mean(metrics_fewshot["test_loss_outer"])

        if jnp.isnan(valid_loss_outer_mean):
            raise ValueError("Loss diverged to nan.")

        # Bandit
        agent_state, metrics_bandit = run_bandit(self.config_bandit, hparams)

        total_regret_norm = np.sum(metrics_bandit["norm_regret"], axis=1)
        total_regret_norm_per_delta_mean = {
            delta: np.mean(total_regret_norm[np.array(self.config_bandit["deltas"]) == delta])
            for delta in np.unique(self.config_bandit["deltas"])
        }

        return {
            "valid_loss_outer_mean": valid_loss_outer_mean.item(),
            "test_loss_outer_mean": test_loss_outer_mean.item(),
            **{
                "norm_regret_delta{}_mean".format(int(100 * delta)): val.item()
                for delta, val in total_regret_norm_per_delta_mean.items()
            },

        }


# Load the configuration
user_config = parse_arguments(sys.argv[1:])
cfg = load_default_config(user_config["method"], user_config["meta_model"])

# Set logging directory
name = time.strftime("%Y%m%d_%H%M%S") + "_ray_metabandit_{}_{}".format(
    cfg["num_trials"], cfg["method"]
)
local_dir = os.path.join(os.getcwd(), "logs", "ray")
log_dir = os.path.join(local_dir, name)
print("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))

analysis = tune.run(
    MetaBandit,
    config=cfg,
    metric="norm_regret_delta95_mean",
    mode="min",
    name=name,
    num_samples=1,
    local_dir=local_dir,
    raise_on_failed_trial=False,
    resources_per_trial={"cpu": 1, "gpu": 1.0},
    stop={"training_iteration": 0},
    verbose=1
)

# Save the results, best run and search space
analysis.results_df.to_csv(os.path.join(log_dir, name + "_results.csv"), index=False, float_format='%.8f')
data.save_dict_as_json(analysis.best_config, name + "_best_config", log_dir)
data.save_dict_as_json(cfg, name + "_search_space", log_dir)
