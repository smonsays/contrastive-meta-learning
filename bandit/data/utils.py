
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

from .base import Dataset, ExtendedDataset


def batch_generator(rng, inputs, targets, batch_size, steps):
    """
    Get set of random batches from data.
    """
    rng = jax.random.split(rng, steps)
    batch_get_batch = jax.vmap(get_batch, in_axes=(0, None, None, None))

    return batch_get_batch(rng, inputs, targets, batch_size)


def create_metadataset(inputs, targets, shots, meta_batch_size):
    """
    Split data into train and test set and create batches of tasks on leading axis.

    Args:
        inputs: jnp.array with shape (num_tasks, num_samples, input_dim)
        targets: jnp.array with shape (num_tasks, num_samples, output_dim)
        shots: Number of samples used for train (support) set
        meta_batch_size: number of tasks per meta batch
    """
    # Split into train shots and use remaining as test shots
    inputs_train, inputs_test = jnp.split(inputs, indices_or_sections=(shots, ), axis=1)
    targets_train, targets_test = jnp.split(targets, indices_or_sections=(shots, ), axis=1)

    if meta_batch_size is not None:
        # Create batches over tasks
        num_meta_batches = inputs.shape[0] // meta_batch_size

        def reshape_to_batches(x):
            return jnp.reshape(
                x, (num_meta_batches, meta_batch_size, *x.shape[1:])
            )

        inputs_train = reshape_to_batches(inputs_train)
        targets_train = reshape_to_batches(targets_train)
        inputs_test = reshape_to_batches(inputs_test)
        targets_test = reshape_to_batches(targets_test)

    return ExtendedDataset(inputs_train, targets_train, inputs_test, targets_test)


def get_batch(rng, inputs, targets, batch_size):
    """
    Get single random batch from data.
    """
    # Draw random indeces with replacement
    idx = jax.random.choice(rng, inputs.shape[0], (batch_size, ), replace=True)

    return Dataset(inputs[idx], targets[idx])
