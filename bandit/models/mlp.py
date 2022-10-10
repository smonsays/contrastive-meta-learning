"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MultilayerPerceptron(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.readout = nn.Dense(self.output_dim, kernel_init=self.kernel_init)
        self.body = FeedforwardNetwork(self.hidden_dims, self.activation, self.kernel_init)

    def __call__(self, x):
        x = self.readout(self.body(x))
        return x

    def features(self, x):
        return self.body(x)


class FeedforwardNetwork(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.hidden_layers = [
            nn.Dense(h_dim, kernel_init=self.kernel_init,)
            for h_dim in self.hidden_dims
        ]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = jnp.reshape(x, (x.shape[0], -1))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def forward_modulated(self, x, bias, gain):
        x = jnp.reshape(x, (x.shape[0], -1))
        for l, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x) * gain[l] + bias[l])

        return x
