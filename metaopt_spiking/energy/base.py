"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc

import torch


class EnergyFunction(torch.nn.Module, abc.ABC):
    """
    Abstract energy function class.
    """
    # TODO: All losses implicitly perform reduction='mean', maybe this should be exposed to make it explicit.
    def __init__(self):
        super().__init__()

    def __add__(self, other):
        return EnergySum([self, other])

    @abc.abstractmethod
    def forward(self, output, target, meta_model, state):
        pass


class EnergySum(EnergyFunction):
    def __init__(self, energy_list):
        super().__init__()
        self.energy_modules = torch.nn.ModuleList(energy_list)

    def forward(self, output, target, meta_model, state):
        loss = 0
        for module in self.energy_modules:
            loss += module(output, target, meta_model, state)

        return loss
