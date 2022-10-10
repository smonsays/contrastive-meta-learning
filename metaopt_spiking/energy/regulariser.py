"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from .base import EnergyFunction


class ActivityNeuron(EnergyFunction):
    """
    Activity regularisation at the neuron level to avoid quiescent neurons as in Zenke (2021).
    """
    def __init__(self, lamda=None, nu=None):
        super().__init__()
        self.lamda = lamda
        self.nu = nu

    def forward(self, output, target, meta_model, state):
        if all(key in meta_model.meta_learner for key in ["activity_neuron_lambda", "activity_neuron_nu"]):
            loss = [
                l * torch.relu(nu - torch.sum(s, dim=1))**2
                for s, l, nu in zip(
                    state["spiking"],
                    meta_model.meta_learner["activity_neuron_lambda"],
                    meta_model.meta_learner["activity_neuron_nu"])
            ]
        else:
            loss = [
                self.lamda * torch.relu(self.nu - torch.sum(s, dim=1))**2
                for s in state["spiking"]
            ]

        return torch.mean(torch.cat(loss))


class ActivityPopulation(EnergyFunction):
    """
    Activity regularisation at the population level to limit average activity as in Zenke (2021).
    """
    def __init__(self, l_p=2, lamda=None, nu=None):
        """
        Args:
            l_p: order of regularisation used (usually L1 or L2)
        """
        super().__init__()
        self.l_p = l_p
        self.lamda = lamda
        self.nu = nu

        self.lamda_key = "activity_population_lambda_l{}".format(str(l_p))
        self.nu_key = "activity_population_nu_l{}".format(str(l_p))

    def forward(self, output, target, meta_model, state):
        if all(key in meta_model.meta_learner for key in [self.lamda_key, self.nu_key]):
            loss = [
                l * torch.relu(torch.mean(torch.sum(s, dim=1), dim=1) - nu)**self.l_p
                for s, l, nu in zip(
                    state["spiking"],
                    meta_model.meta_learner[self.lamda_key],
                    meta_model.meta_learner[self.nu_key])
            ]
        else:
            loss = [
                self.lamda * torch.relu(torch.mean(torch.sum(s, dim=1), dim=1) - self.nu)**self.l_p
                for s in state["spiking"]
            ]

        return torch.mean(torch.cat(loss))


class MeanFiringRate(EnergyFunction):
    """
    Activity regularisation at the population level as in Bellec (2018).
    """
    def __init__(self, target, strength):
        super().__init__()
        self.target = target
        self.strength = strength

    def forward(self, output, target, meta_model, state):
        loss = [
            (self.strength * (torch.mean(s) - self.target)**2).view(-1)
            for s in state["spiking"]
        ]

        return torch.mean(torch.cat(loss))


class ElasticRegularizer(EnergyFunction):
    def __init__(self, l2_strength=None):
        super().__init__()
        self.l2_strength = l2_strength

    def forward(self, output, target, meta_model, state):
        if "l2_strength" in meta_model.meta_learner:
            loss = [
                (torch.exp(lamda) * (omega - p)**2).view(-1)
                for p, lamda, omega in zip(
                    meta_model.base_parameters(),
                    meta_model.meta_learner["l2_strength"].parameters(),
                    meta_model.meta_learner["omega"].parameters()
                )
            ]
        else:
            loss = [
                (self.l2_strength * (omega - p)**2).view(-1)
                for p, omega in zip(
                    meta_model.base_parameters(),
                    meta_model.meta_learner["omega"].parameters()
                )
            ]

        return torch.sum(torch.cat(loss))


class L2Regularizer(EnergyFunction):
    def __init__(self, l2=None):
        super().__init__()
        self.l2 = l2

    def forward(self, output, target, meta_model, state):
        if "l2" in meta_model.meta_learner:
            # If hyperparams contain l2 strengths, use them
            loss = [
                (l2 * p**2).view(-1)
                for p, l2 in zip(meta_model.base_parameters(), meta_model.meta_learner["l2"])
            ]
        else:
            # Otherwise use default l2 strength defined upon init
            loss = [
                (self.l2 * p**2).view(-1)
                for p in meta_model.base_parameters()
            ]

        return torch.sum(torch.cat(loss))


class L1Spikes(EnergyFunction):
    """
    L1 loss on total number of spikes.
    """
    def forward(self, output, target, meta_model, state):
        loss = [
            l1 * torch.mean(torch.sum(s, dim=(1, 2)))
            for s, l1 in zip(state["spiking"], meta_model.meta_learner["l1_spike"])
        ]

        return torch.mean(torch.stack(loss))


class L2Spikes(EnergyFunction):
    """
    L2 loss on spikes per neuron.
    """
    def forward(self, output, target, meta_model, state):
        loss = [
            l2 * torch.mean(torch.sum(s, dim=1)**2)
            for s, l2 in zip(state["spiking"], meta_model.meta_learner["l2_spike"].parameters())
        ]

        return torch.mean(torch.stack(loss))
