"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os

import numpy as np

from .dataset import sinusoid, wheel

from .base import DATAPATH
from .io import load_pytree, save_pytree
from .utils import create_metadataset


def load_metadataset(
    name,
    shots_train,
    shots_test,
    num_tasks_train,
    num_tasks_test,
    num_tasks_valid,
    meta_batch_size,
    repeat_train_tasks=1,
    load_from_disk=False,
    save_to_disk=False,
    seed=None,
):

    rng = np.random.default_rng(seed)
    num_samples = shots_train + shots_test
    path = os.path.join(
        DATAPATH,
        "{}_shots-{}-{}_tasks-{}-{}-{}_seed-{}.pickle".format(
            name, shots_train, shots_test, num_tasks_train, num_tasks_test, num_tasks_valid, seed
        ),
    )

    if load_from_disk:
        data_dict = load_pytree(path)
        inputs_train, targets_train = data_dict["train"]
        inputs_test, targets_test = data_dict["test"]
        inputs_valid, targets_valid = data_dict["valid"]

    else:
        if name == "sinusoid":
            inputs_train, targets_train = sinusoid.sample_tasks(num_tasks_train, num_samples, rng)
            inputs_test, targets_test = sinusoid.sample_tasks(num_tasks_test, num_samples, rng)
            inputs_valid, targets_valid = sinusoid.sample_tasks(num_tasks_valid, num_samples, rng)

        elif name == "wheel":
            inputs_train, targets_train = wheel.sample_tasks(num_tasks_train, num_samples, rng)
            inputs_test, targets_test = wheel.sample_tasks(num_tasks_test, num_samples, rng)
            inputs_valid, targets_valid = wheel.sample_tasks(num_tasks_valid, num_samples, rng)

        data_dict = {
            "train": (inputs_train, targets_train),
            "test": (inputs_test, targets_test),
            "valid": (inputs_valid, targets_valid),
        }
        if save_to_disk:
            save_pytree(path, data_dict, overwrite=True)

    if repeat_train_tasks > 1:
        idx = rng.choice(num_tasks_train, (num_tasks_train * repeat_train_tasks, ), replace=True)
        inputs_train, targets_train = inputs_train[idx], targets_train[idx]

    metatrainset = create_metadataset(inputs_train, targets_train, shots_train, meta_batch_size)
    metatestset = create_metadataset(inputs_test, targets_test, shots_train, None)
    metavalidset = create_metadataset(inputs_valid, targets_valid, shots_train, None)

    return metatrainset, metatestset, metavalidset
