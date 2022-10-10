"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math

import torch

from .layers.neuron import LeakyLinear


class SpikingNetwork(torch.nn.Module):
    """
    Feedforward spiking network with LIF-neurons and current-based exponential synapses.
    NOTE: Resting potential hard-coded to 0, spike threshold hard-coded 1
    """
    def __init__(self, dimensions, tau, step_size=1):
        """
        Args:
            dimensions: list with [num_input, num_hidden, num_readout]
            tau: dict with time constants τ for "membrane", "synapse", "readout" in ms
            step_size: Δt in ms
        """
        super().__init__()
        assert all(key in tau for key in ["membrane", "synapse"])

        self.beta = {key: float(math.exp(-step_size / t)) for key, t in tau.items()}
        self.hidden_layers = torch.nn.ModuleList(
            LeakyLinear(dim1, dim2, self.beta["membrane"], self.beta["synapse"])
            for dim1, dim2 in zip(dimensions[:-2], dimensions[1:-1])
        )
        # HACK: Get a weight matrix with the pytorch standard initialisation
        self.readout_weight = torch.nn.Linear(dimensions[-2], dimensions[-1], bias=False).weight

    def forward(self, x):
        state = {key: [] for key in ["current", "spiking", "voltage"]}

        # Hidden layer
        for hidden_layer in self.hidden_layers:
            x, voltage, current = hidden_layer(x)

            state["current"].append(current)
            state["voltage"].append(voltage)
            state["spiking"].append(x)

        # Readout layer
        readout = torch.einsum("bti, ji -> btj", x, self.readout_weight)

        # Output average firing of each readout unit over time
        output = torch.mean(readout, dim=1)

        return output, state
