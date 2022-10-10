
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
import optax

from flax.core.frozen_dict import FrozenDict
from optax import TransformInitFn

import energy
import meta
import models
import utils

from . import replay
from .replay import Transition, ReplayBufferState
from .base import BanditAlgorithm


@flax.struct.dataclass
class NeuralEpsilonState:
    params_net: FrozenDict
    hparams: FrozenDict
    optim_net: TransformInitFn
    buffer_state: ReplayBufferState
    t: int


class NeuralEpsilon(BanditAlgorithm):
    def __init__(
        self,
        num_actions,
        context_dim,
        buffer_size,
        epsilon,
        meta_model,
        lr,
        batch_size,
        num_updates,
        optimizer,
        reset_optim,
        reset_params,
        train_freq,
    ):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.meta_model = meta_model
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.reset_optim = reset_optim
        self.reset_params = reset_params
        self.train_freq = train_freq
        self.buffer = replay.ReplayBuffer()
        self.optimizer = utils.create_optimizer(optimizer, {"learning_rate": lr})

    def reset(self, rng, hparams):
        rng_net, rng_hparam = jax.random.split(rng, 2)

        # Initialise neural network base
        input_net = jnp.empty((self.batch_size, self.context_dim))
        if hparams is None:
            hparams = self.meta_model.reset_hparams(rng_hparam, input_net)

        params_net = self.meta_model.reset_params(rng_net, hparams, input_net)
        optim_net = self.optimizer.init(params_net)

        # Initialise the replay buffer
        buffer_state = self.buffer.reset(
            self.context_dim, self.context_dim, self.buffer_size
        )

        return NeuralEpsilonState(params_net, hparams, optim_net, buffer_state, 0)

    def act(self, rng, state, context):
        """
        Act by epsilon-greedily selecting the action with the highest reward.
        """
        # Predict best action
        predicted_rewards = self.meta_model(
            params=state.params_net,
            hparams=state.hparams,
            x=jnp.expand_dims(context, 0),
        ).squeeze()
        best_action = jnp.argmax(predicted_rewards)

        # Add epsilon/num_action probability to each action and remaining mass to optimal action
        probs = jnp.ones((self.num_actions,)) * self.epsilon / self.num_actions
        probs = probs.at[best_action].add(1.0 - self.epsilon)

        return jax.random.categorical(rng, jnp.log(probs))

    def update(self, rng, state, context, action, reward):
        """
        Update the parameters of the Bayesian linear regression and the neural network.
        """
        # Add data to buffers (NOTE: there is no embedding, so we simply store the context twice)
        buffer_state = self.buffer.add(
            state.buffer_state, Transition(context, context, action, reward)
        )

        # Update the neural network base every `train_freq` steps
        def empty_update(rng, params_net, hparams, optim_net, buffer_state):
            empty_metrics = {
                "loss": -jnp.ones((self.num_updates)),
                "gradnorm": -jnp.ones((self.num_updates))
            }
            return params_net, optim_net, buffer_state, empty_metrics

        params_net, optim_net, buffer_state, metrics = jax.lax.cond(
            state.t % self.train_freq,
            empty_update,
            self._update_net,
            rng, state.params_net, state.hparams, state.optim_net, buffer_state
        )

        updated_state = NeuralEpsilonState(
            params_net, state.hparams, optim_net, buffer_state, state.t + 1
        )

        return updated_state, metrics

    def _update_net(self, rng, params_net, hparams, optim_net, buffer_state):
        # Retrain the network
        params_net, optim_net, metrics = self.train_net(
            rng, params_net, hparams, optim_net, buffer_state
        )

        return params_net, optim_net, buffer_state, metrics

    def train_net(self, rng, params, hparams, optim, buffer_state):
        """
        Train the neural network on the replay buffer.
        """
        rng_data, rng_net = jax.random.split(rng, 2)

        # Sample a dataset from the replay buffer
        # NOTE: When the buffer is not filled sufficiently, this will contain many repeated samples
        dataset_sample = self.buffer.sample_dataset(
            rng_data, buffer_state, self.batch_size, self.num_updates
        )
        # Reset parameters and optimizer state
        if self.reset_params:
            params = self.meta_model.reset_params(rng_net, hparams, dataset_sample.context[0])

        if self.reset_optim:
            optim = self.optimizer.init(params)

        def loss_fn(params, hparams, contexts, actions, rewards):
            # Create targets containing rewards and actions
            actions_one_hot = jax.nn.one_hot(actions, self.num_actions)
            rewards_one_hot = jnp.expand_dims(rewards, -1) * actions_one_hot

            targets = jnp.concatenate(
                (jnp.expand_dims(rewards_one_hot, -2), jnp.expand_dims(actions_one_hot, -2)),
                axis=-2
            )
            pred = self.meta_model(params, hparams, contexts)

            return self.meta_model.loss_fn_inner(pred, targets, params, hparams)

        def train_step(state, batch):
            params, hparams, optim = state

            loss, grads = jax.value_and_grad(loss_fn)(
                params, hparams, batch.context, batch.action, batch.reward
            )
            params_update, optim = self.optimizer.update(grads, optim, params)
            params = optax.apply_updates(params, params_update)

            metrics = {
                "loss": loss,
                "gradnorm": optax.global_norm(grads)
            }

            return [params, hparams, optim], metrics

        carry, metrics = jax.lax.scan(train_step, [params, hparams, optim], dataset_sample)
        params, hparams, optim = carry

        return params, optim, metrics


@flax.struct.dataclass
class NeuralThompsonState:
    params_linear: FrozenDict
    params_net: FrozenDict
    hparams: FrozenDict
    optim_net: TransformInitFn
    buffer_state: ReplayBufferState
    t: int


class NeuralThompson(BanditAlgorithm):
    def __init__(
        self,
        num_actions,
        context_dim,
        buffer_size,
        a0,
        b0,
        lambda_prior,
        hidden_dims,
        lr,
        lr_decay_rate,
        l2_reg,
        batch_size,
        num_updates,
        optimizer,
        max_grad_norm,
        reset_optim,
        reset_params,
        train_freq,
    ):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.feature_dim = hidden_dims[-1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.reset_optim = reset_optim
        self.reset_params = reset_params
        self.train_freq = train_freq
        self.buffer = replay.ReplayBuffer()

        # Setup the forward models
        self.linear = models.BayesianLinearRegression(self.feature_dim, a0, b0, lambda_prior, False)
        self.meta_model = meta.module.ComplexSynapse(
            loss_fn_inner=energy.squared_error_masked(reduction="mean"),
            loss_fn_outer=None,
            # NOTE: Bandit showdown uses scaled uniform init for MLP
            base_learner=models.MultilayerPerceptron(hidden_dims, num_actions),
            l2_reg=l2_reg,
            variant="imaml",
        )

        # Setup optimizer with inverse time decay lr schedule
        def inverse_time_decay(step):
            return lr / (1.0 + lr_decay_rate * step)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            utils.create_optimizer(optimizer, {"learning_rate": inverse_time_decay})
        )

        # Vmap linear head functions for the ensemble of heads
        self.heads_fit = jax.jit(jax.vmap(partial(self.linear.apply, method=self.linear.fit)))
        self.heads_predict = jax.jit(jax.vmap(self.linear.apply, in_axes=(0, 0, None)))
        self.heads_init = jax.vmap(self.linear.init)

    def reset(self, rng, hparams):
        rng_net, rng_hparam, rng_linear = jax.random.split(rng, 3)

        # Initialise neural network base
        input_net = jnp.empty((self.batch_size, self.context_dim))
        if hparams is None:
            hparams = self.meta_model.reset_hparams(rng_hparam, input_net)

        params_net = self.meta_model.reset_params(rng_net, hparams, input_net)
        optim_net = self.optimizer.init(params_net)

        # Initialise Bayesian linear heads (one per action)
        rngs_linear = jax.random.split(rng_linear, self.num_actions)
        inputs_linear = jnp.empty((self.num_actions, self.feature_dim))
        params_linear = self.heads_init(rngs_linear, rngs_linear, inputs_linear)

        # Initialise the replay buffer
        buffer_state = self.buffer.reset(
            self.context_dim, self.feature_dim, self.buffer_size
        )

        return NeuralThompsonState(params_linear, params_net, hparams, optim_net, buffer_state, 0)

    def act(self, rng, state, context):
        """
        Act by Thompson sampling from the linear head given the encoding from the neural network.
        """
        # Compute encoding for the current context
        encoding = self.meta_model(
            params=state.params_net,
            hparams=state.hparams,
            x=jnp.expand_dims(context, 0),
            method=self.meta_model.base_learner.features
        ).squeeze()

        # Thompson sampling using Bayesian linear regression values
        rng_thompson = jax.random.split(rng, self.num_actions)
        values = self.heads_predict(state.params_linear, rng_thompson, encoding)

        return jnp.argmax(values)

    def update(self, rng, state, context, action, reward):
        """
        Update the parameters of the Bayesian linear regression and the neural network.
        """
        # Add data to buffers
        encoding = self.meta_model(
            params=state.params_net,
            hparams=state.hparams,
            x=jnp.expand_dims(context, 0),
            method=self.meta_model.base_learner.features
        ).squeeze()
        buffer_state = self.buffer.add(state.buffer_state, Transition(context, encoding, action, reward))

        # Update the neural network base every `train_freq` steps
        def empty_update(rng, params_net, hparams, optim_net, buffer_state):
            empty_metrics = {
                "loss": -jnp.ones((self.num_updates)),
                "gradnorm": -jnp.ones((self.num_updates))
            }
            return params_net, optim_net, buffer_state, empty_metrics

        params_net, optim_net, buffer_state, metrics = jax.lax.cond(
            state.t % self.train_freq,
            empty_update,
            self._update_net,
            rng, state.params_net, state.hparams, state.optim_net, buffer_state
        )

        # Update the linear heads
        params_linear = self._update_linear(state.params_linear, buffer_state)

        updated_state = NeuralThompsonState(
            params_linear, params_net, state.hparams, optim_net, buffer_state, state.t + 1
        )

        return updated_state, metrics

    def _update_linear(self, params_linear, buffer_state):
        """
        Update params of Bayesian linear heads
        """
        # Mask encodings and rewards based on actions
        mask_per_action = jax.nn.one_hot(buffer_state.actions, self.num_actions).T
        batch_hadamard = jax.vmap(jnp.multiply, in_axes=(0, None))
        encodings_masked = batch_hadamard(
            jnp.expand_dims(mask_per_action, -1), buffer_state.encodings
        )
        rewards_masked = batch_hadamard(mask_per_action, buffer_state.rewards)

        # Fit all Bayesian linear regression heads
        params_linear = self.heads_fit(params_linear, encodings_masked, rewards_masked)

        return params_linear

    def _update_net(self, rng, params_net, hparams, optim_net, buffer_state):
        # Retrain the network
        params_net, optim_net, metrics = self.train_net(
            rng, params_net, hparams, optim_net, buffer_state
        )

        # Update the encodings of every datapoint collected so far
        encodings_new = self.meta_model(
            params=params_net,
            hparams=hparams,
            x=buffer_state.contexts,
            method=self.meta_model.base_learner.features
        )
        # NOTE: This also updates empty contexts
        buffer_state = buffer_state.replace(encodings=encodings_new)

        return params_net, optim_net, buffer_state, metrics

    def train_net(self, rng, params, hparams, optim, buffer_state):
        """
        Train the neural network on the replay buffer.
        """
        rng_data, rng_net = jax.random.split(rng, 2)

        # Sample a dataset from the replay buffer
        # NOTE: When the buffer is not filled sufficiently, this will contain many repeated samples
        dataset_sample = self.buffer.sample_dataset(
            rng_data, buffer_state, self.batch_size, self.num_updates
        )
        # Reset parameters and optimizer state
        if self.reset_params:
            params = self.meta_model.reset_params(rng_net, hparams, dataset_sample.context[0])

        if self.reset_optim:
            optim = self.optimizer.init(params)

        def loss_fn(params, hparams, contexts, actions, rewards):
            # Create targets containing rewards and actions
            actions_one_hot = jax.nn.one_hot(actions, self.num_actions)
            rewards_one_hot = jnp.expand_dims(rewards, -1) * actions_one_hot

            targets = jnp.concatenate(
                (jnp.expand_dims(rewards_one_hot, -2), jnp.expand_dims(actions_one_hot, -2)),
                axis=-2
            )
            pred = self.meta_model(params, hparams, contexts)

            return self.meta_model.loss_fn_inner(pred, targets, params, hparams)

        def train_step(state, batch):
            params, hparams, optim = state

            loss, grads = jax.value_and_grad(loss_fn)(
                params, hparams, batch.context, batch.action, batch.reward
            )
            params_update, optim = self.optimizer.update(grads, optim, params)
            params = optax.apply_updates(params, params_update)

            metrics = {
                "loss": loss,
                "gradnorm": optax.global_norm(grads)
            }

            return [params, hparams, optim], metrics

        carry, metrics = jax.lax.scan(train_step, [params, hparams, optim], dataset_sample)
        params, hparams, optim = carry

        return params, optim, metrics
