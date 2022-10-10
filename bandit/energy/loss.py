"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import jax
import jax.numpy as jnp
import optax

from .utils import reduce


def cross_entropy(reduction="mean", axis=None):
    def loss_fn(output, target, params, hparams):
        target_one_hot = jax.nn.one_hot(target, output.shape[-1])
        loss = optax.softmax_cross_entropy(output, target_one_hot)
        return reduce(loss, reduction, axis)

    return loss_fn


def squared_error_masked(reduction="mean", axis=None):
    def loss_fn(output, target, params, hparams):
        reward = jnp.take(target, 0, axis=-2)
        mask = jnp.take(target, 1, axis=-2)

        loss = (output - reward)**2
        loss_masked = jnp.sum(mask * loss, axis=-1)

        return reduce(loss_masked, reduction, axis)

    return loss_fn


def squared_error(reduction="mean", axis=None):
    def loss_fn(output, target, params, hparams):
        loss = (output - target)**2
        return reduce(loss, reduction, axis)

    return loss_fn
