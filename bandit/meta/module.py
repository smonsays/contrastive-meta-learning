"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import NamedTuple

import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.core.frozen_dict import FrozenDict

import energy
import models


class MetaModule:
    def __init__(self):
        pass

    def __call__(self, meta_state, x, method=None):
        pass

    def reset_hparams(self, rng, sample_input):
        pass

    def reset_params(self, rng, hparams, sample_input):
        pass


class ComplexSynapseMetaParams(NamedTuple):
    omega: FrozenDict
    log_lambda: FrozenDict


class ComplexSynapseParams(NamedTuple):
    base_learner: FrozenDict


class ComplexSynapse(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, base_learner, l2_reg=None, variant="imaml"):
        self.loss_fn_inner = loss_fn_inner
        self.loss_fn_outer = loss_fn_outer

        self.base_learner = base_learner
        self.l2_reg = l2_reg
        self.variant = variant

        if variant == "complex_synapse":
            assert l2_reg is not None
            self.loss_fn_inner = energy.add(self.loss_fn_inner, energy.complex_synapse(l2_reg))

        elif variant == "imaml":
            assert l2_reg is not None
            self.loss_fn_inner = energy.add(self.loss_fn_inner, energy.imaml(l2_reg))

        elif variant == "init":
            self.loss_fn_inner = self.loss_fn_inner

        elif variant == "l2_reg":
            self.loss_fn_inner = energy.add(self.loss_fn_inner, energy.l2_learned())

        else:
            raise ValueError("Variant \"{}\" not defined.".format(variant))

    def __call__(self, params, hparams, x, method=None):
        return self.base_learner.apply({"params": params.base_learner}, x, method=method)

    def reset_hparams(self, rng, sample_input):
        hparams_omega = self.base_learner.init(rng, sample_input)["params"]

        if self.variant in ["complex_synapse", "l2_reg"]:
            hparams_log_lambda = jtu.tree_map(
                lambda x: jnp.log(self.l2_reg) * jnp.ones_like(x), hparams_omega
            )
        else:
            hparams_log_lambda = FrozenDict()

        return ComplexSynapseMetaParams(hparams_omega, hparams_log_lambda)

    def reset_params(self, rng, hparams, sample_input):
        return ComplexSynapseParams(hparams.omega)


class GainModMetaParams(NamedTuple):
    body: FrozenDict
    head_init: FrozenDict
    bias_init: FrozenDict
    gain_init: FrozenDict


class GainModParams(NamedTuple):
    head: FrozenDict
    bias: FrozenDict
    gain: FrozenDict


class GainMod(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, hidden_dims, output_dim):
        self.loss_fn_inner = loss_fn_inner
        self.loss_fn_outer = loss_fn_outer

        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.body = models.FeedforwardNetwork(hidden_dims)
        self.head = flax.linen.Dense(output_dim)

    def __call__(self, params, hparams, x):

        features = self.body.apply(
            {"params": hparams.body}, x, params.bias, params.gain,
            method=self.body.forward_modulated
        )

        return self.head.apply({"params": params.head}, features)

    def reset_hparams(self, rng, sample_input):
        rng_body, rng_head = jax.random.split(rng, 2)

        hparams_body = self.body.init(rng_body, sample_input)["params"]
        hparams_head = self.head.init(
            rng_head, jnp.empty((sample_input.shape[0], self.hidden_dims[0]))
        )["params"]

        hparams_bias = FrozenDict({
            l: jnp.zeros(h_dim) for l, h_dim in enumerate(self.hidden_dims)
        })
        hparams_gain = FrozenDict({
            l: jnp.ones(h_dim) for l, h_dim in enumerate(self.hidden_dims)
        })

        return GainModMetaParams(hparams_body, hparams_head, hparams_bias, hparams_gain)

    def reset_params(self, rng, hparams, sample_input):
        params_head = hparams.head_init

        return GainModParams(params_head, hparams.bias_init, hparams.gain_init)


class AlmostNoInnerLoopMetaParams(NamedTuple):
    body: FrozenDict
    head_init: FrozenDict


class AlmostNoInnerLoopParams(NamedTuple):
    head: FrozenDict


class AlmostNoInnerLoop(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, hidden_dims, output_dim):
        self.loss_fn_inner = loss_fn_inner
        self.loss_fn_outer = loss_fn_outer

        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.body = models.FeedforwardNetwork(hidden_dims)
        self.head = flax.linen.Dense(output_dim)

    def __call__(self, params, hparams, x):

        features = self.body.apply({"params": hparams.body}, x)

        return self.head.apply({"params": params.head}, features)

    def reset_hparams(self, rng, sample_input):
        hparams_body = self.body.init(rng, sample_input)["params"]
        hparams_head = self.head.init(rng, jnp.empty((1, self.hidden_dims[0])))["params"]

        return AlmostNoInnerLoopMetaParams(hparams_body, hparams_head)

    def reset_params(self, rng, hparams, sample_input):
        params_head = hparams.head_init

        return AlmostNoInnerLoopParams(params_head)
