"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import higher
import torch


def infinite_iterator(iterable):
    """
    Converts an iterable into an infinitely long iterable.
    """
    if len(iterable) == 1:
        # HACK: If the iterable has a single entry, save time by caching it.
        # This is only correct if the train transforms are deterministic
        item = next(iter(iterable))
        while True:
            yield item
    else:
        while True:
            for item in iterable:
                yield item


def make_optimizer_differentiable(optimizer, params):
    if isinstance(optimizer, torch.optim.Adam):
        optimizer_diff = higher.optim.DifferentiableAdam(optimizer, params)

    elif isinstance(optimizer, torch.optim.Adamax):
        optimizer_diff = higher.optim.DifferentiableAdamax(optimizer, params)

    elif isinstance(optimizer, torch.optim.SGD):
        optimizer_diff = higher.optim.DifferentiableSGD(optimizer, params)

    else:
        raise ValueError("Optimizer \"{}\" undefined".format(optimizer))

    return optimizer_diff
