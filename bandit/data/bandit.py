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
class ContextualBanditState:
    contexts: jnp.ndarray
    rewards: jnp.ndarray
    regrets: jnp.ndarray
    order: jnp.ndarray


class ContextualBandit:

    def __init__(self, num_actions, num_contexts, context_dim):
        self.num_actions = num_actions
        self.num_contexts = num_contexts
        self.context_dim = context_dim

    def reset(self, rng, contexts, rewards, regrets):
        """
        Randomly shuffle the order of the contexts to deliver.
        """
        order = jax.random.permutation(rng, self.num_contexts)
        return ContextualBanditState(contexts, rewards, regrets, order)

    def context(self, state, idx):
        """
        Returns the number-th context.
        """
        return state.contexts[state.order[idx]]

    def regret(self, state, idx, action):
        """
        Returns the regret for the idx-th context and action.
        """
        return state.regrets[state.order[idx]][action]

    def step(self, state, idx, action):
        """
        Returns the reward for the idx-th context and action.
        """
        return state.rewards[state.order[idx]][action]
