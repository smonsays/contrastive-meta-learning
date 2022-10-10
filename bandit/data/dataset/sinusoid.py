"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np


def sample_data(num_samples, amplitude, phase, seed=None):
    rng = np.random.default_rng(seed)

    inputs = rng.uniform(low=-5., high=5., size=(num_samples, 1))
    targets = amplitude * np.sin(inputs + phase)

    return inputs, targets


def sample_tasks(num_tasks, num_samples, seed=None):
    rng = np.random.default_rng(seed)

    amplitudes = rng.uniform(low=0.1, high=0.5, size=(num_tasks, ))
    phases = rng.uniform(low=0.0, high=np.pi, size=(num_tasks, ))

    inputs, targets = zip(*[
        sample_data(num_samples, amp, phase, rng) for amp, phase in zip(amplitudes, phases)
    ])

    return np.stack(inputs), np.stack(targets)


if __name__ == "__main__":
    # NOTE: Run using python -m data.sinusoid
    import matplotlib.pyplot as plt

    inputs, targets = sample_tasks(num_tasks := 3, num_samples := 100)

    for i in range(num_tasks):
        plt.scatter(inputs[i], targets[i])

    plt.show()
