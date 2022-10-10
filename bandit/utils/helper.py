"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import jax.numpy as jnp
import optax


def create_optimizer(name, kwargs={}):
    """
    Create optax.GradientTransformation. Kwargs can be partially
    defined with remaining parameters set as defaults.
    """
    if name == "adam":
        # Could be more concise in python 3.9 kwargs = {...} | kwargs
        defaults = {"learning_rate": 0.001, "eps_root": jnp.finfo(jnp.float32).eps}
        defaults.update(kwargs)
        return optax.adam(**defaults)

    elif name == "adamw":
        # Could be more concise in python 3.9 kwargs = {...} | kwargs
        defaults = {"learning_rate": 0.001, "eps_root": jnp.finfo(jnp.float32).eps}
        defaults.update(kwargs)
        return optax.adamw(**defaults)

    elif name == "rmsprop":
        defaults = {"learning_rate": 0.01}
        defaults.update(kwargs)
        return optax.rmsprop(**defaults)

    elif name == "sgd":
        defaults = {"learning_rate": 0.01}
        defaults.update(kwargs)
        return optax.sgd(**defaults)

    elif name == "sgd_nesterov":
        defaults = {"learning_rate": 0.01, "momentum": 0.9, "nesterov": True}
        defaults.update(kwargs)
        return optax.sgd(**defaults)

    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))
