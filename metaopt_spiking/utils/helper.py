"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch


def create_nonlinearity(name):
    """
    Create nonlinearity function given its name
    """
    if name == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif name == "linear":
        class PassThrough(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return input
        return PassThrough()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    else:
        raise ValueError("Nonlinearity \"{}\" undefined".format(name))


def create_optimizer(name, parameters, kwargs={}):
    """
    Create torch.optim.Optimizer. Kwargs can be partially defined
    with remaining parameters set as defaults.
    """
    if name == "adam":
        # Could be more concise in python 3.9 kwargs = {...} | kwargs
        defaults = {"lr": 0.001, "betas": (0.9, 0.999)}
        defaults.update(kwargs)
        return torch.optim.Adam(parameters, **defaults)

    elif name == "adamax":
        defaults = {"lr": 0.002, "betas": (0.9, 0.999)}
        defaults.update(kwargs)
        return torch.optim.Adamax(parameters, **defaults)
    elif name == "rmsprop":
        defaults = {"lr": 0.01}
        defaults.update(kwargs)
        return torch.optim.RMSprop(parameters, **defaults)

    elif name == "sgd":
        defaults = {"lr": 0.01}
        defaults.update(kwargs)
        return torch.optim.SGD(parameters, **defaults)

    elif name == "sgd_nesterov":
        defaults = {"lr": 0.01, "momentum": 0.9, "nesterov": True}
        defaults.update(kwargs)
        return torch.optim.SGD(parameters, **defaults)

    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))


def create_scheduler(name, optimizer, kwargs={}):
    """
    Create torch.optim.LRscheduler given its name
    """
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, kwargs.get("step", 500), kwargs.get("decay", 0.1))
    elif name == "multiplicative":
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda _: kwargs.get("decay", 0.9999))
    elif name == "delayed_target":
        def create_schedule(decay, step, end):
            """
            Creates a custom schedule that anneals the lr towards a target value after a certain number of steps.

            Args:
                decay: target decay at the end of the schedule
                step: steps after which to start the decaying
                end: steps after which decay ends
            """
            # Choose gamma such that we reach target decay after (end - step) multiplications with lr_start
            gamma = (decay)**(1 / (end - step))

            def schedule(epoch):
                if epoch >= step:
                    return max(decay, gamma**(epoch - step + 1))
                else:
                    return 1.0

            return schedule

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=create_schedule(kwargs.get("decay", 0.1), kwargs.get("step", 2000), kwargs.get("end", 3000)))
    else:
        raise ValueError("Scheduler \"{}\" undefined".format(name))
