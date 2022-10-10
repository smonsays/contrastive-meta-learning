"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json
import logging
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import agents
import data
import energy
import meta
import models
import utils


class AggregateStats(NamedTuple):
    mean: float
    sem: float


def load_default_config(agent):
    """
    Load default parameter configuration.

    Returns:
        Dictionary of default parameters for the given task
    """
    if agent == "NeuralLinear":
        return {
            "a0": 12.0,
            "agent": "NeuralLinear",
            "b0": 30.0,
            "batch_size": 512,
            "buffer_size": 80000,
            "context_dim": 2,
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "hidden_dims": [100, 100],
            "initial_pulls": 2,
            "l2_reg": 0.0,
            "lambda_prior": 23.0,
            "lr": 0.01,
            "lr_decay_rate": 0.5,
            "max_grad_norm": 5.0,
            "num_actions": 5,
            "num_trials": 80000,
            "num_updates": 50,
            "optimizer": "rmsprop",
            "reset_optim": True,
            "reset_params": False,
            "seed": 2022,
            "train_freq": 20,
        }
    elif agent == "NeuralGreedy":
        return {
            "agent": "NeuralGreedy",
            "batch_size": 512,
            "buffer_size": 2000,
            "context_dim": 2,
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "epsilon": 0.0,
            "hidden_dims": [100, 100],
            "initial_pulls": 2,
            "l2_reg": 0.0,
            "lr": 0.01,
            "model": "gain_mod",
            "num_actions": 5,
            "num_trials": 2000,
            "num_updates": 50,
            "optimizer": "sgd",
            "reset_optim": False,
            "reset_params": False,
            "seed": 2022,
            "train_freq": 20,
        }
    elif agent == "LinFullPost":
        return {
            "a0": 6.0,
            "agent": "LinFullPost",
            "b0": 6.0,
            "buffer_size": 80000,
            "context_dim": 2,
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "initial_pulls": 2,
            "lambda_prior": 0.25,
            "num_actions": 5,
            "num_trials": 80000,
            "seed": 2022,
        }
    elif agent == "LinGreedy":
        return {
            "agent": "LinGreedy",
            "buffer_size": 2000,
            "context_dim": 2,
            "deltas": [0.5, 0.7, 0.9, 0.95, 0.99],
            "epsilon": 0.05,
            "initial_pulls": 2,
            "l2_reg": 0.25,
            "num_actions": 5,
            "num_trials": 2000,
            "seed": 2022,
        }
    elif agent == "Uniform":
        return {
            "agent": "Uniform",
            "buffer_size": 2000,
            "context_dim": 2,
            "deltas": [0.5, 0.5, 0.7, 0.9, 0.95, 0.99],
            "initial_pulls": 2,
            "num_actions": 5,
            "num_trials": 2000,
            "seed": 2022,
        }
    else:
        raise ValueError("Default configuration for agent \"{}\" not defined.".format(agent))


def run_episode(rng, env, env_state, agent, agent_state, trials, initial_pulls):
    """
    Run a full episode and return cumulative regret.
    """
    @jax.jit
    def bandit_step(carry, t):
        """
        lax.scan compatible bandit step.
        """
        rng, agent_state = carry
        rng, rng_action, rng_update = jax.random.split(rng, 3)

        # Get the new context
        context = env.context(env_state, t)

        # Round robin until each action has been selected `initial_pulls` times
        action = jax.lax.cond(
            t < env.num_actions * initial_pulls,
            lambda rng, state, context: t % env.num_actions,
            agent.act,  # afterwards perform agent action
            rng_action, agent_state, context
        )

        # Act in the environment and collect reward
        reward = env.step(env_state, t, action)
        regret = env.regret(env_state, t, action)

        # Update the agent's state
        agent_state, metrics = agent.update(rng_update, agent_state, context, action, reward)

        carry = [rng, agent_state]
        metrics = {
            "context": context,
            "action": action,
            "regret": regret,
            "reward": reward,
            **metrics
        }

        return carry, metrics

    carry, metrics = jax.lax.scan(bandit_step, [rng, agent_state], jnp.arange(trials))
    _, agent_state = carry

    return agent_state, metrics


def run_bandit(cfg, hparams=None, raytune=False):

    rng_np = np.random.default_rng(cfg["seed"])
    rng = jax.random.PRNGKey(cfg["seed"])
    rng_env, rng_agent, rng_episode = jax.random.split(rng, 3)

    # Create the contextual bandit environment
    env = data.ContextualBandit(
        num_actions=cfg["num_actions"],
        num_contexts=cfg["num_trials"],
        context_dim=cfg["context_dim"],
    )

    # Generate data for each delta of the bandit task
    wheel_data = zip(*[
        data.wheel.sample_data(num_contexts=cfg["num_trials"], delta=delta, seed=rng_np)
        for delta in cfg["deltas"]
    ])
    contexts, rewards, regrets = [jnp.stack(d) for d in wheel_data]

    # Initialise the environment states
    rng_env = jax.random.split(rng_env, len(cfg["deltas"]))
    env_states = jax.vmap(env.reset)(rng_env, contexts, rewards, regrets)

    # Create the bandit agent
    if cfg["agent"] == "Uniform":
        agent = agents.Uniform(cfg["num_actions"])

    elif cfg["agent"] == "LinGreedy":
        agent = agents.LinearEpsilon(
            num_actions=cfg["num_actions"],
            context_dim=cfg["context_dim"],
            buffer_size=cfg["buffer_size"],
            epsilon=cfg["epsilon"],
            l2_reg=cfg["l2_reg"],
        )
    elif cfg["agent"] == "LinFullPost":
        # NOTE: We need double precision to match the bandit showdown numbers
        from jax.config import config
        config.update("jax_enable_x64", True)

        agent = agents.LinearThompson(
            num_actions=cfg["num_actions"],
            context_dim=cfg["context_dim"],
            buffer_size=cfg["buffer_size"],
            a0=cfg["a0"],
            b0=cfg["b0"],
            lambda_prior=cfg["lambda_prior"],
        )
    elif cfg["agent"] == "NeuralGreedy":

        loss_fn = energy.squared_error_masked(reduction="mean")

        if cfg["model"] == "imaml":
            meta_model = meta.module.ComplexSynapse(
                loss_fn_inner=loss_fn,
                loss_fn_outer=None,
                base_learner=models.MultilayerPerceptron(
                    cfg["hidden_dims"], output_dim=cfg["num_actions"]
                ),
                l2_reg=cfg["l2_reg"],
                variant="imaml"
            )
        elif cfg["model"] == "gain_mod":
            meta_model = meta.module.GainMod(
                loss_fn_inner=loss_fn,
                loss_fn_outer=None,
                hidden_dims=cfg["hidden_dims"],
                output_dim=cfg["num_actions"],
            )
        elif cfg["model"] == "learned_init":
            meta_model = meta.module.ComplexSynapse(
                loss_fn_inner=loss_fn,
                loss_fn_outer=None,
                base_learner=models.MultilayerPerceptron(
                    cfg["hidden_dims"], output_dim=cfg["num_actions"]
                ),
                l2_reg=None,
                variant="init"
            )
        else:
            raise ValueError("Model \"{}\" not defined.".format(cfg["model"]))

        agent = agents.NeuralEpsilon(
            num_actions=cfg["num_actions"],
            context_dim=cfg["context_dim"],
            buffer_size=cfg["buffer_size"],
            epsilon=cfg["epsilon"],
            meta_model=meta_model,
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            num_updates=cfg["num_updates"],
            optimizer=cfg["optimizer"],
            reset_optim=cfg["reset_optim"],
            reset_params=cfg["reset_params"],
            train_freq=cfg["train_freq"],
        )
    elif cfg["agent"] == "NeuralLinear":
        agent = agents.NeuralThompson(
            num_actions=cfg["num_actions"],
            context_dim=cfg["context_dim"],
            buffer_size=cfg["buffer_size"],
            a0=cfg["a0"],
            b0=cfg["b0"],
            lambda_prior=cfg["lambda_prior"],
            hidden_dims=cfg["hidden_dims"],
            lr_decay_rate=cfg["lr_decay_rate"],
            lr=cfg["lr"],
            l2_reg=cfg["l2_reg"],
            batch_size=cfg["batch_size"],
            num_updates=cfg["num_updates"],
            optimizer=cfg["optimizer"],
            max_grad_norm=cfg["max_grad_norm"],
            reset_optim=cfg["reset_optim"],
            reset_params=cfg["reset_params"],
            train_freq=cfg["train_freq"],
        )
    else:
        raise ValueError("Agent \"{}\" not defined.".format(cfg["agent"]))

    agent_state = agent.reset(rng_agent, hparams)

    # Run the episodes
    batch_run_episode = jax.vmap(run_episode, in_axes=(0, None, 0, None, None, None, None))
    rng_episode = jax.random.split(rng_episode, len(cfg["deltas"]))
    agent_state, metrics = batch_run_episode(
        rng_episode, env, env_states, agent, agent_state, cfg["num_trials"], cfg["initial_pulls"]
    )

    # Logging

    # Compute (normalised) regret (per delta)
    norm_regret = np.array([
        reg / data.wheel.uniform_regret(delta, cfg["num_trials"])
        for reg, delta in zip(metrics["regret"], cfg["deltas"])
    ])
    metrics["norm_regret"] = norm_regret

    total_regret = np.sum(metrics["regret"], axis=1)
    total_regret_per_delta = {
        delta: total_regret[np.array(cfg["deltas"]) == delta]
        for delta in np.unique(cfg["deltas"])
    }
    total_regret_norm = np.sum(norm_regret, axis=1)
    total_regret_norm_per_delta = {
        delta: total_regret_norm[np.array(cfg["deltas"]) == delta]
        for delta in np.unique(cfg["deltas"])
    }

    # Aggregated results across deltas
    total_regret_stats = AggregateStats(
        np.mean(total_regret),
        np.std(total_regret) / np.sqrt(metrics["regret"].shape[0])
    )

    total_regret_norm_stats = AggregateStats(
        np.mean(total_regret_norm),
        np.std(total_regret_norm) / np.sqrt(metrics["regret"].shape[0])
    )

    # Aggregated results per deltas
    total_regret_norm_per_delta_stats = {
        delta: AggregateStats(
            np.mean(values),
            np.std(values) / np.sqrt(values.shape[0]),
        )
        for delta, values in total_regret_norm_per_delta.items()
    }

    total_regret_per_delta_stats = {
        delta: AggregateStats(
            np.mean(values),
            np.std(values) / np.sqrt(values.shape[0]),
        )
        for delta, values in total_regret_per_delta.items()
    }

    if raytune:
        return {
            "total_regret_mean": total_regret_stats.mean,
            "norm_regret_mean": total_regret_norm_stats.mean,
            **{
                "norm_regret_delta{}_mean".format(int(100 * delta)): stats.mean
                for delta, stats in total_regret_norm_per_delta_stats.items()
            },
        }
    else:
        logging.info("total_regret: {:.2f}±{:.2f}".format(total_regret_stats.mean, total_regret_stats.sem))
        logging.info("norm_regret: {:.2f}±{:.2f}".format(total_regret_norm_stats.mean, total_regret_norm_stats.sem))

        for delta, stats in total_regret_norm_per_delta_stats.items():
            logging.info("norm_regret_delta{}: {:.2f}±{:.2f}".format(
                int(100 * delta), stats.mean, stats.sem)
            )

        for delta, stats in total_regret_per_delta_stats.items():
            logging.info("total_regret_delta{}: {:.2f}±{:.2f}".format(
                int(100 * delta), stats.mean, stats.sem)
            )

        return agent_state, metrics


if __name__ == "__main__":
    cfg = load_default_config("Uniform")

    # Setup logger
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_bandit_wheel_{}_{}".format(
        cfg["num_trials"], cfg["agent"]
    )
    log_dir = utils.setup_logging(run_id)

    logging.info("Running on {}".format(jax.default_backend()))
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))

    # with jax.disable_jit():
    agent_state, metrics = run_bandit(cfg)

    # Save results, model and configuration to disk
    data.save_pytree(os.path.join(log_dir, run_id + "_metrics"), metrics)
    data.save_pytree(os.path.join(log_dir, run_id + "_model"), agent_state)
    data.save_dict_as_json(cfg, run_id + "_config", log_dir)

    # Additional data processing for one of the repetitons
    import pandas as pd
    df = pd.DataFrame({
        "action": metrics["action"][0],
        "reward": metrics["reward"][0],
        "regret": metrics["regret"][0]
    })
    df["cumregret_per_action"] = df.groupby("action")["regret"].cumsum()
    df["optimal_action"] = (df["regret"] == 0)
    df["cum_action"] = df.groupby("action")["action"].cumcount()
    df["cum_optimal_action"] = df.groupby("action")["optimal_action"].cumsum()

    # Visualise loss
    import matplotlib.pyplot as plt

    if "regret" in metrics:
        cum_regret = jnp.cumsum(metrics["regret"], axis=1)
        cum_regret_mean = jnp.mean(cum_regret, axis=0)
        cum_regret_std = jnp.std(cum_regret, axis=0)

        # Cumulative regret
        fig, ax = plt.subplots(figsize=[6.0, 3.0])
        ax.plot(x := range(len(cum_regret_mean)), cum_regret_mean)
        ax.fill_between(x, cum_regret_mean - cum_regret_std, cum_regret_mean + cum_regret_std, alpha=.3)
        fig.savefig(os.path.join(log_dir, run_id + "_cumregret.pdf"))

        fig, ax = plt.subplots(figsize=[6.0, 3.0])
        df.groupby('action')['cumregret_per_action'].plot(ax=ax, legend=True)
        fig.savefig(os.path.join(log_dir, run_id + "_cumregret_per_action.pdf"))

    if "action" in metrics:
        # Histogram of actions
        fig, ax = plt.subplots(figsize=[6.0, 3.0])
        ax.hist(np.array(metrics["action"][0]), density=True, bins=cfg["num_actions"])
        fig.savefig(os.path.join(log_dir, run_id + "_actions_hist.pdf"))

        # Actions over time colour-coded by optimality (single repetition)
        fig, ax = plt.subplots(figsize=[12.0, 2.0])
        t = np.arange(len(df))
        ax.scatter(t[df["optimal_action"]], df["action"][df["optimal_action"]], s=0.5)
        ax.scatter(t[~df["optimal_action"]], df["action"][~df["optimal_action"]], color="red", s=0.5)
        fig.savefig(os.path.join(log_dir, run_id + "_actions_scatter.pdf"))

        # Fraction optimal action was chosen when it was optimal per action  (single repetition)
        fig, ax = plt.subplots(figsize=[6.0, 3.0])
        frac_opt = df.groupby("action")["optimal_action"].sum() / df.groupby("action")["action"].count()
        ax.bar(range(len(frac_opt)), frac_opt)
        fig.savefig(os.path.join(log_dir, run_id + "_fraction_optimal_actions.pdf"))

    if "context" in metrics:
        fig, ax = plt.subplots(figsize=[6.0, 6.0])
        circ_outer = plt.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='red')
        circ_inner = plt.Circle((0, 0), radius=cfg["deltas"][0], edgecolor='black', facecolor='blue')
        ax.add_patch(circ_outer)
        ax.add_patch(circ_inner)
        ax.vlines(x=[0.0, 0.0], ymin=[-1.0, cfg["deltas"][0]], ymax=[-cfg["deltas"][0], 1.0], color="black")
        ax.hlines(y=[0.0, 0.0], xmin=[-1.0, cfg["deltas"][0]], xmax=[-cfg["deltas"][0], 1.0], color="black")
        ax.scatter(metrics["context"][0][:, 0], metrics["context"][0][:, 1], color="white", s=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        fig.savefig(os.path.join(log_dir, run_id + "_contexts.pdf"))

    if "loss" in metrics:
        fig, ax = plt.subplots(figsize=[6.0, 3.0])
        ax.plot(metrics["loss"][0][::500].T)
        ax.set_yscale('log')
        fig.savefig(os.path.join(log_dir, run_id + "_loss.pdf"))
