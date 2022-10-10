"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class ReplayBufferState:
    contexts: jnp.ndarray
    encodings: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_slot: int
    full: bool


@flax.struct.dataclass
class Transition:
    context: jnp.ndarray
    encoding: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray


class ReplayBuffer:
    def reset(self, context_dim, encoding_dim, max_size):
        next_slot = 0
        full = False

        contexts = jnp.zeros((max_size, context_dim))
        encodings = jnp.zeros((max_size, encoding_dim))
        actions = jnp.full((max_size, ), -1.0)
        rewards = jnp.zeros((max_size, ))

        return ReplayBufferState(contexts, encodings, actions, rewards, next_slot, full)

    def add(self, buffer, transition):

        contexts = buffer.contexts.at[buffer.next_slot].set(transition.context)
        encodings = buffer.encodings.at[buffer.next_slot].set(transition.encoding)
        actions = buffer.actions.at[buffer.next_slot].set(transition.action)
        rewards = buffer.rewards.at[buffer.next_slot].set(transition.reward)

        # Keep track if buffer as been filled at least once
        max_size = len(contexts)
        full = jax.lax.cond(
            ((buffer.next_slot + 1) == max_size),
            lambda _: True,
            lambda _: buffer.full,
            None,
        )
        # If buffer is filled, start replacing values FIFO
        next_slot = jax.lax.cond(
            (buffer.next_slot + 1) < max_size,
            lambda next_slot: next_slot + 1,
            lambda _: 0,
            buffer.next_slot
        )

        return ReplayBufferState(contexts, encodings, actions, rewards, next_slot, full)

    def sample(self, rng, buffer, batch_size):
        idx = jax.random.randint(rng, (batch_size,), 0, buffer.next_slot)
        batch_transition = Transition(
            buffer.contexts[idx], buffer.encodings[idx], buffer.actions[idx], buffer.rewards[idx]
        )
        return batch_transition

    def sample_dataset(self, rng, buffer, batch_size, num_samples):
        rngs_samples = jax.random.split(rng, num_samples)
        batch_sample = jax.vmap(self.sample, in_axes=(0, None, None))

        return batch_sample(rngs_samples, buffer, batch_size)
