"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import jax.numpy as jnp
import jax.tree_util as jtu

import utils

from .utils import reduce


def complex_synapse(reduction="mean"):
    def complex_synapse_fn(param, omega, log_lambda):
        return jnp.exp(log_lambda) * (param - omega)**2

    def loss_fn(output, target, params, hparams):

        loss_tree = jtu.tree_map(
            complex_synapse_fn, params.base_learner, hparams.omega, hparams.log_lambda
        )
        loss = utils.flatcat(loss_tree)

        return reduce(loss, reduction, axis=None)

    return loss_fn


def l2_learned(reduction="mean"):
    def loss_fn(output, target, params, hparams):
        def l2_learned_fn(param, log_lambda):
            return jnp.exp(log_lambda) * (param)**2

        loss_tree = jtu.tree_map(l2_learned_fn, params.base_learner, hparams.log_lambda)
        loss = utils.flatcat(loss_tree)

        return reduce(loss, reduction, axis=None)

    return loss_fn


def imaml(reg_strength, reduction="mean"):
    def imaml_fn(param, omega):
        return (param - omega)**2

    def loss_fn(output, target, params, hparams):

        loss_tree = jtu.tree_map(imaml_fn, params.base_learner, hparams.omega)
        loss = utils.flatcat(loss_tree)

        return reg_strength * reduce(loss, reduction, axis=None)

    return loss_fn
