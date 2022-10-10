"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import flax
import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict

import models

from . import replay
from .replay import Transition, ReplayBufferState
from .base import BanditAlgorithm


@flax.struct.dataclass
class LinearState:
    params_linear: FrozenDict
    buffer_state: ReplayBufferState


class LinearThompson(BanditAlgorithm):
    def __init__(self, num_actions, context_dim, buffer_size, a0, b0, lambda_prior):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.buffer_size = buffer_size
        self.buffer = replay.ReplayBuffer()

        # Vmap linear head functions for the ensemble of heads
        self.linear = models.BayesianLinearRegression(
            context_dim, a0, b0, lambda_prior, intercept=True
        )
        self.heads_fit = jax.vmap(partial(self.linear.apply, method=self.linear.fit))
        self.heads_predict = jax.vmap(self.linear.apply, in_axes=(0, 0, None))
        self.heads_init = jax.vmap(self.linear.init)

    def reset(self, rng, hparams=None):
        # Initialise Bayesian linear heads (one per action)
        rngs_linear = jax.random.split(rng, self.num_actions)
        inputs_linear = jnp.ones((self.num_actions, self.context_dim))
        params_linear = self.heads_init(rngs_linear, rngs_linear, inputs_linear)

        # Initialise the replay buffer
        buffer_state = self.buffer.reset(self.context_dim, self.context_dim, self.buffer_size)

        return LinearState(params_linear, buffer_state)

    def act(self, rng, state, context):
        """
        Act by Thompson sampling from the linear heads.
        """
        # Thompson sampling using Bayesian linear regression values
        rng_thompson = jax.random.split(rng, self.num_actions)
        values = self.heads_predict(state.params_linear, rng_thompson, context)

        return jnp.argmax(values)

    def update(self, rng, state, context, action, reward):
        """
        Update the parameters of the Bayesian linear regression.
        """
        # Add data to buffers (NOTE: there is no embedding, so we simply store the context twice)
        buffer_state = self.buffer.add(
            state.buffer_state, Transition(context, context, action, reward)
        )

        # Mask contexts and rewards based on actions
        mask_per_action = jax.nn.one_hot(buffer_state.actions, self.num_actions)
        batch_hadamard = jax.vmap(jnp.multiply, in_axes=(1, None))

        contexts_masked = batch_hadamard(
            jnp.expand_dims(mask_per_action, -1), buffer_state.contexts
        )
        rewards_masked = batch_hadamard(mask_per_action, buffer_state.rewards)

        # Update all Bayesian linear regression heads
        params_linear = self.heads_fit(state.params_linear, contexts_masked, rewards_masked)

        return LinearState(params_linear, buffer_state), {}


class LinearEpsilon(BanditAlgorithm):
    def __init__(self, num_actions, context_dim, buffer_size, epsilon, l2_reg):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.buffer = replay.ReplayBuffer()

        # Vmap linear head functions for the ensemble of heads
        self.linear = models.RidgeRegression(context_dim, l2_reg, intercept=True)
        self.heads_fit = jax.vmap(partial(self.linear.apply, method=self.linear.fit))
        self.heads_predict = jax.vmap(self.linear.apply, in_axes=(0, None))
        self.heads_init = jax.vmap(self.linear.init)

    def reset(self, rng, hparams=None):
        # Initialise linear heads (one per action)
        rngs_linear = jax.random.split(rng, self.num_actions)
        inputs_linear = jnp.ones((self.num_actions, self.context_dim))
        params_linear = self.heads_init(rngs_linear, inputs_linear)

        # Initialise the replay buffer
        buffer_state = self.buffer.reset(self.context_dim, self.context_dim, self.buffer_size)

        return LinearState(params_linear, buffer_state)

    def act(self, rng, state, context):
        """
        Act epsilon-greedly wrt to the linear heads.
        """
        # Predict best action
        values = self.heads_predict(state.params_linear, context)
        best_action = jnp.argmax(values)

        # Add epsilon/num_action probability to each action and remaining mass to optimal action
        probs = jnp.ones((self.num_actions,)) * self.epsilon / self.num_actions
        probs = probs.at[best_action].add(1.0 - self.epsilon)

        return jax.random.categorical(rng, jnp.log(probs))

    def update(self, rng, state, context, action, reward):
        """
        Update the parameters of the linear regression.
        # TODO: This is basically the same as in Bayesian linear
        """
        # Add data to buffers (NOTE: there is no embedding, so we simply store the context twice)
        buffer_state = self.buffer.add(
            state.buffer_state, Transition(context, context, action, reward)
        )

        # Mask contexts and rewards based on actions
        mask_per_action = jax.nn.one_hot(buffer_state.actions, self.num_actions)
        batch_hadamard = jax.vmap(jnp.multiply, in_axes=(1, None))

        contexts_masked = batch_hadamard(
            jnp.expand_dims(mask_per_action, -1), buffer_state.contexts
        )
        rewards_masked = batch_hadamard(mask_per_action, buffer_state.rewards)

        # Update all linear regression heads
        params_linear = self.heads_fit(state.params_linear, contexts_masked, rewards_masked)

        return LinearState(params_linear, buffer_state), {}
