"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
import logging
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
import tensorboardX

import data
import energy
import meta
import models
import utils


def load_default_config(method, meta_model, dataset):
    """
    Load default parameter configuration.

    Returns:
        Dictionary of default parameters for the given task
    """
    if method == "cml" and meta_model == "gain_mod" and dataset == "sinusoid":
        return {
            "batch_size": 1,
            "beta": 1.0,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "lr_inner": 0.001,
            "lr_nudged": 0.0001,
            "lr_outer": 0.03,
            "meta_batch_size": 25,
            "method": "cml",
            "meta_model": "gain_mod",
            "optim_inner": "sgd",
            "optim_outer": "adamw",
            "seed": 2022,
            "steps_inner": 1000,
            "steps_nudged": 100
        }

    elif method == "cml" and meta_model == "gain_mod" and dataset == "wheel":
        return {
            "batch_size": 100,
            "beta": 10.0,
            "dataset": {
                "name": "wheel",
                "num_tasks_test": 10,
                "num_tasks_train": 64,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 500,
                "shots_test": 50,
                "shots_train": 512
            },
            "hidden_dims": [100, 100],
            "lr_inner": 0.003,
            "lr_nudged": 0.003,
            "lr_outer": 0.03,
            "meta_batch_size": 8,
            "meta_model": "gain_mod",
            "method": "cml",
            "optim_inner": "adam",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 1000,
            "steps_nudged": 100
        }
    elif method == "cml" and meta_model == "imaml" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "beta": 3.0,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "l2_reg": 10.0,
            "lr_inner": 0.01,
            "lr_nudged": 0.01,
            "lr_outer": 0.03,
            "meta_batch_size": 25,
            "method": "cml",
            "meta_model": "imaml",
            "optim_inner": "adam",
            "optim_outer": "adamw",
            "seed": 2022,
            "steps_inner": 500,
            "steps_nudged": 250
        }
    elif method == "cml" and meta_model == "imaml" and dataset == "wheel":
        return {
            "batch_size": 512,
            "beta": 0.3,
            "dataset": {
                "name": "wheel",
                "num_tasks_test": 10,
                "num_tasks_train": 64,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 100,
                "shots_test": 50,
                "shots_train": 512
            },
            "hidden_dims": [100, 100],
            "l2_reg": 1000.0,
            "lr_inner": 0.0001,
            "lr_nudged": 0.03,
            "lr_outer": 0.03,
            "meta_batch_size": 8,
            "method": "cml",
            "meta_model": "imaml",
            "optim_inner": "adam",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 250,
            "steps_nudged": 100
        }

    elif method == "cml" and meta_model == "learned_l2" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "beta": 1.0,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "l2_reg": 10.0,
            "lr_inner": 0.01,
            "lr_nudged": 0.0003,
            "lr_outer": 0.1,
            "meta_batch_size": 25,
            "meta_model": "learned_l2",
            "method": "cml",
            "optim_inner": "adam",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 1000,
            "steps_nudged": 1000
        }

    elif method == "maml" and meta_model == "anil" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "lr_inner": 0.001,
            "lr_outer": 0.001,
            "meta_batch_size": 25,
            "meta_model": "anil",
            "method": "maml",
            "optim_inner": "sgd",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 10,
        }

    elif method == "maml" and meta_model == "gain_mod" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "lr_inner": 0.001,
            "lr_outer": 0.001,
            "meta_batch_size": 25,
            "meta_model": "gain_mod",
            "method": "maml",
            "optim_inner": "sgd",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 10,
        }
    elif method == "maml" and meta_model == "gain_mod" and dataset == "wheel":
        return {
            "batch_size": 512,
            "dataset": {
                "name": "wheel",
                "num_tasks_train": 64,
                "num_tasks_test": 10,
                "num_tasks_valid": 10,
                "repeat_train_tasks": 100,
                "shots_train": 512,
                "shots_test": 50,
            },
            "hidden_dims": [100, 100],
            "lr_inner": 0.001,
            "lr_outer": 0.01,
            "meta_batch_size": 8,
            "meta_model": "gain_mod",
            "method": "maml",
            "optim_inner": "sgd",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 100,
        }

    elif method == "maml" and meta_model == "imaml" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "l2_reg": 1.0,
            "lr_inner": 0.001,
            "lr_outer": 0.001,
            "meta_batch_size": 25,
            "meta_model": "imaml",
            "method": "maml",
            "optim_inner": "sgd",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 10,
        }

    elif method == "maml" and meta_model == "learned_init" and dataset == "sinusoid":
        return {
            "batch_size": 10,
            "dataset": {
                "name": "sinusoid",
                "num_tasks_test": 100,
                "num_tasks_train": 25000,
                "num_tasks_valid": 100,
                "repeat_train_tasks": 1,
                "shots_test": 10,
                "shots_train": 10
            },
            "hidden_dims": [40, 40],
            "lr_inner": 0.001,
            "lr_outer": 0.001,
            "meta_batch_size": 25,
            "meta_model": "learned_init",
            "method": "maml",
            "optim_inner": "sgd",
            "optim_outer": "adam",
            "seed": 2022,
            "steps_inner": 10,
        }
    else:
        raise ValueError("Default configuration for method \"{}\" not defined.".format(method))


def make_meta_train(meta_learner, optim_fn_outer):
    meta_test_fn = make_meta_test(meta_learner)

    @jax.jit
    def meta_train(rng, hparams, metatrainset, metavalidset):
        """
        Meta-training.
        """
        def train_step(state, meta_batch):
            rng, hparams, meta_optim = state
            rng, rng_batch, rng_valid = jax.random.split(rng, 3)

            # Evaluate hypergrad over all tasks in meta_batch in parallel
            rng_batch = jax.random.split(rng_batch, len(meta_batch.x_train))
            batch_grad = jax.vmap(meta_learner.grad, in_axes=(0, None, 0))
            hgrads, metrics_train = batch_grad(rng_batch, hparams, meta_batch)

            # Average hypergrad across tasks and update
            hgrads = jtu.tree_map(partial(jnp.mean, axis=0), hgrads)
            hparams_update, meta_optim = optim_fn_outer.update(hgrads, meta_optim, hparams)
            hparams = optax.apply_updates(hparams, hparams_update)

            # Evaluate on validation set
            metrics_valid = meta_test_fn(rng_valid, hparams, metavalidset)

            # Combine metrics
            metrics_train = utils.prepend_keys(metrics_train, "train")
            metrics_valid = utils.prepend_keys(metrics_valid, "valid")
            metrics = {**metrics_train, **metrics_valid}

            return [rng, hparams, meta_optim], metrics

        optim_outer_init = optim_fn_outer.init(hparams)
        carry, metrics = jax.lax.scan(train_step, [rng, hparams, optim_outer_init], metatrainset)
        _, hparams, _ = carry

        return hparams, metrics

    return meta_train


def make_meta_test(meta_learner):

    @jax.jit
    def meta_test(rng, hparams, metatestset):
        batch_eval = jax.vmap(meta_learner.eval, in_axes=(0, None, 0))
        rng_eval = jax.random.split(rng, len(metatestset.x_train))
        loss, metrics_inner = batch_eval(rng_eval, hparams, metatestset)

        return {"loss_outer": loss, **metrics_inner}

    return meta_test


def run_fewshot(cfg, hparams=None, raytune=False, writer=None):

    # Prepare random number generators
    rng = jax.random.PRNGKey(cfg["seed"])
    rng_init, rng_log, rng_train, rng_test = jax.random.split(rng, 4)

    # Create metadatasets
    metatrainset, metatestset, metavalidset = data.load_metadataset(
        name=cfg["dataset"]["name"],
        shots_train=cfg["dataset"]["shots_train"],
        shots_test=cfg["dataset"]["shots_test"],
        num_tasks_train=cfg["dataset"]["num_tasks_train"],
        num_tasks_test=cfg["dataset"]["num_tasks_test"],
        num_tasks_valid=cfg["dataset"]["num_tasks_valid"],
        meta_batch_size=cfg["meta_batch_size"],
        repeat_train_tasks=cfg["dataset"]["repeat_train_tasks"],
        load_from_disk=False,
        seed=cfg["seed"],
    )

    # Create loss functions and meta-model
    if cfg["dataset"]["name"] == "wheel":
        loss_fn_inner = energy.squared_error_masked(reduction="mean")
        loss_fn_outer = energy.squared_error_masked(reduction="mean")
    else:
        loss_fn_inner = energy.squared_error(reduction="sum")
        loss_fn_outer = energy.squared_error(reduction="sum")

    if cfg["meta_model"] == "anil":
        meta_model = meta.module.AlmostNoInnerLoop(
            loss_fn_inner, loss_fn_outer, cfg["hidden_dims"], metatrainset.y_train.shape[-1],
        )
    elif cfg["meta_model"] == "complex_synapse":
        net = models.MultilayerPerceptron(
            cfg["hidden_dims"], output_dim=metatrainset.y_train.shape[-1]
        )
        meta_model = meta.module.ComplexSynapse(
            loss_fn_inner, loss_fn_outer, net, l2_reg=cfg["l2_reg"], variant="complex_synapse"
        )
    elif cfg["meta_model"] == "gain_mod":
        meta_model = meta.module.GainMod(
            loss_fn_inner, loss_fn_outer, cfg["hidden_dims"],
            metatrainset.y_train.shape[-1]
        )
    elif cfg["meta_model"] == "imaml":
        net = models.MultilayerPerceptron(
            cfg["hidden_dims"], output_dim=metatrainset.y_train.shape[-1]
        )
        meta_model = meta.module.ComplexSynapse(
            loss_fn_inner, loss_fn_outer, net, l2_reg=cfg["l2_reg"], variant="imaml"
        )
    elif cfg["meta_model"] == "learned_init":
        net = models.MultilayerPerceptron(
            cfg["hidden_dims"], output_dim=metatrainset.y_train.shape[-1]
        )
        meta_model = meta.module.ComplexSynapse(
            loss_fn_inner, loss_fn_outer, net, l2_reg=None, variant="init"
        )
    elif cfg["meta_model"] == "learned_l2":
        net = models.MultilayerPerceptron(
            cfg["hidden_dims"], output_dim=metatrainset.y_train.shape[-1]
        )
        meta_model = meta.module.ComplexSynapse(
            loss_fn_inner, loss_fn_outer, net, l2_reg=cfg["l2_reg"], variant="l2_reg"
        )
    else:
        raise ValueError("Model \"{}\" not defined.".format(cfg["meta_model"]))

    # Create optimisers
    optim_fn_outer = utils.create_optimizer(cfg["optim_outer"], {"learning_rate": cfg["lr_outer"]})
    optim_fn_inner = utils.create_optimizer(cfg["optim_inner"], {"learning_rate": cfg["lr_inner"]})

    # Setup the meta-learning algorithm
    if cfg["method"] == "maml":
        meta_learner = meta.learner.MAML(
            meta_model, optim_fn_inner,
            cfg["batch_size"], cfg["steps_inner"],
        )
    elif cfg["method"] == "cg":
        meta_learner = meta.learner.ConjugateGradient(
            meta_model, optim_fn_inner,
            cfg["batch_size"], cfg["steps_inner"], cfg["steps_cg"]
        )
    elif cfg["method"] == "cml":
        optim_fn_nudged = utils.create_optimizer(
            cfg["optim_inner"], {"learning_rate": cfg["lr_nudged"]}
        )
        meta_learner = meta.learner.EquilibriumPropagation(
            meta_model, optim_fn_inner, optim_fn_nudged,
            cfg["batch_size"], cfg["steps_inner"], cfg["steps_nudged"], cfg["beta"]
        )
    elif cfg["method"] == "cml_symmetric":
        optim_fn_nudged = utils.create_optimizer(
            cfg["optim_inner"], {"learning_rate": cfg["lr_nudged"]}
        )
        meta_learner = meta.learner.SymmetricEquilibriumPropagation(
            meta_model, optim_fn_inner, optim_fn_nudged,
            cfg["batch_size"], cfg["steps_inner"], cfg["steps_nudged"], cfg["beta"]
        )
    else:
        raise ValueError("Method \"{}\" not defined.".format(cfg["method"]))

    # Meta-training and testing
    meta_train_fn = make_meta_train(meta_learner, optim_fn_outer)
    meta_test_fn = make_meta_test(meta_learner)

    if hparams is None:
        hparams_init = meta_model.reset_hparams(rng_init, metatrainset.x_train[0][0])
    else:
        hparams_init = hparams

    hparams, metrics_train = meta_train_fn(rng_train, hparams_init, metatrainset, metavalidset)
    metrics_test = meta_test_fn(rng_test, hparams, metatestset)

    # Combine metrics
    metrics_test = utils.prepend_keys(metrics_test, "test")
    metrics = {**metrics_train, **metrics_test}

    # Logging
    valid_loss_outer_mean = jnp.mean(metrics["valid_loss_outer"][-1])
    valid_loss_outer_stdv = jnp.std(metrics["valid_loss_outer"][-1])
    test_loss_outer_mean = jnp.mean(metrics["test_loss_outer"])
    test_loss_outer_stdv = jnp.std(metrics["test_loss_outer"])

    # Check for nans in metrics
    anynans = jnp.any(utils.flatcat(jtu.tree_map(lambda x: jnp.any(jnp.isnan(x)), metrics)))

    if not (writer is None or anynans):
        for group in hparams_init._asdict():
            # Hyperparameter histogram before meta-learning
            for (key, val) in utils.flatten_dict(getattr(hparams_init, group).unfreeze()).items():
                writer.add_histogram("hparam/{}/{}".format(group, key), val.reshape(-1), 0)

            # Hyperparameter histogram after meta-learning
            for (key, val) in utils.flatten_dict(getattr(hparams, group).unfreeze()).items():
                writer.add_histogram("hparam/{}/{}".format(group, key), val.reshape(-1), 1)

        for t, val in enumerate(jnp.mean(metrics["valid_loss_outer"], axis=1)):
            writer.add_scalar("valid/loss_outer", val, t)

        if "train_loss_outer" in metrics.keys():
            for t, val in enumerate(jnp.mean(metrics["train_loss_outer"], axis=1)):
                writer.add_scalar("train/loss_outer", val, t)

        for t, val in enumerate(jnp.mean(metrics["train_gradnorm_outer"], axis=1)):
            writer.add_scalar("train/gradnorm_outer", val, t)

        every_n = len(metrics["train_loss_inner"]) // 10
        for t, val in enumerate(jnp.mean(metrics["train_loss_inner"], axis=1)[::every_n].T):
            writer.add_scalars("train/loss_inner", {str(step): v for step, v in enumerate(val)}, t)

        for t, val in enumerate(jnp.mean(metrics["train_gradnorm_inner"], axis=1)[::every_n].T):
            writer.add_scalars(
                "train/gradnorm_inner", {str(step): v for step, v in enumerate(val)}, t
            )

        if "train_loss_nudged" in metrics.keys():
            every_n = len(metrics["train_loss_nudged"]) // 10
            for t, val in enumerate(jnp.mean(metrics["train_loss_nudged"], axis=1)[::every_n].T):
                writer.add_scalars(
                    "train/loss_nudged", {str(step): v for step, v in enumerate(val)}, t
                )

            for t, val in enumerate(jnp.mean(metrics["train_gradnorm_nudged"], axis=1)[::every_n].T):
                writer.add_scalars(
                    "train/gradnorm_nudged", {str(step): v for step, v in enumerate(val)}, t
                )

            if "train_loss_nudged_neg" in metrics.keys():
                for t, val in enumerate(jnp.mean(metrics["train_loss_nudged_neg"], axis=1)[::every_n].T):
                    writer.add_scalars(
                        "train/loss_nudged_neg", {str(step): v for step, v in enumerate(val)}, t
                    )

                for t, val in enumerate(jnp.mean(metrics["train_gradnorm_nudged_neg"], axis=1)[::every_n].T):
                    writer.add_scalars(
                        "train/gradnorm_nudged_neg", {str(step): v for step, v in enumerate(val)}, t
                    )

        # Parameters histogram before/after adaptation to specific task
        # rng_adapt, rng_init = jax.random.split(rng_log)
        # params_init = meta_model.reset_params(rng_init, hparams, metatestset.x_train[0][0])
        # params, _ = meta_learner.adapt(rng_adapt, params_init, hparams, metatestset.x_train[0], metatestset.y_train[0])
        # for group in params_init._asdict():
        #     for (key, val) in utils.flatten_dict(getattr(params_init, group).unfreeze()).items():
        #         writer.add_histogram("param/{}/{}".format(group, key), val.reshape(-1), 0)

        #     for (key, val) in utils.flatten_dict(getattr(params, group).unfreeze()).items():
        #         writer.add_histogram("param/{}/{}".format(group, key), val.reshape(-1), 1)

        # # Sample task adaptation
        # pred = meta_model(params, hparams, x := jnp.linspace(-5, 5))
        # prior = meta_model(params_init, hparams, x := jnp.linspace(-5, 5))
        # fig, ax = plt.subplots()
        # ax.scatter(metatestset.x_train[0], metatestset.y_train[0], label="train")
        # ax.scatter(metatestset.x_test[0], metatestset.y_test[0], label="test")
        # ax.plot(x, pred, label="prediction")
        # ax.plot(x, prior, label="prior")
        # ax.legend()
        # writer.add_figure("sample_task", fig)

    if raytune:
        return {
            "valid_loss_outer": valid_loss_outer_mean.item(),
            "test_loss_outer": test_loss_outer_mean.item(),
        }

    else:
        logging.info("valid_loss_outer: {:.4f}±{:.4f}".format(
            valid_loss_outer_mean, valid_loss_outer_stdv))
        logging.info("test_loss_outer: {:.4f}±{:.4f}".format(
            test_loss_outer_mean, test_loss_outer_stdv))

        return hparams, metrics


if __name__ == "__main__":

    cfg = load_default_config(method="cml", meta_model="gain_mod", dataset="wheel")

    # Setup logger
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_fewshot_{}_{}_{}".format(
        cfg["method"], cfg["meta_model"], cfg["dataset"]["name"]
    )
    log_dir = utils.setup_logging(run_id)
    path_tensorboard = os.path.join(log_dir, run_id + "_tensorboard")
    writer = tensorboardX.SummaryWriter(path_tensorboard)

    # Start the actual run
    logging.info("Running on {}".format(jax.default_backend()))
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))

    # from jax.config import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    hparams, metrics = run_fewshot(cfg, writer=writer)

    # Save results, model state and configuration to disk
    data.save_pytree(os.path.join(log_dir, run_id + "_metrics"), metrics)
    data.save_pytree(os.path.join(log_dir, run_id + "_model"), hparams)
    data.save_dict_as_json(cfg, run_id + "_config", log_dir)

    # Zip the tensorboard logging results and remove the folder to save space
    writer.close()
    utils.zip_and_remove(path_tensorboard)

    # Plot some logs (redundant with tensorboard)
    fig, ax = plt.subplots(figsize=[6.0, 3.0])
    ax.plot(jnp.sum(metrics["valid_loss_outer"], axis=1))
    fig.savefig(os.path.join(log_dir, run_id + "_valid-loss-outer.pdf"))

    fig, ax = plt.subplots(figsize=[6.0, 3.0])
    ax.plot(jnp.sum(metrics["train_loss_inner"], axis=1)[::100].T)
    fig.savefig(os.path.join(log_dir, run_id + "_train-loss-inner.pdf"))
