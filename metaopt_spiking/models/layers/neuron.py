"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from .surrogate import HeavisideFastSigmoid


class LeakyLinear(torch.nn.Linear):
    """
    Linear layer with LIF neurons.

    NOTE: This approach of combining the feedforward input operation (nn.Linear)
          with the spiking activation (LIF) allows to slightly speed up the forward
          propagation using a single einsum opearation over all time steps and track
          the dynamic variables over time inside the layer. However, a more modular
          approach that would allow to use arbitrary feedforward operations (e.g. convolutions)
          would perform the feedforward mapping outside this class and only process
          a single time-step inside the LIFNeuron.
    """
    def __init__(self, input_dim, output_dim, beta_mem, beta_syn):
        super().__init__(input_dim, output_dim, bias=False)

        self.beta_mem = beta_mem
        self.beta_syn = beta_syn

        # State
        self.register_buffer("current", None)
        self.register_buffer("voltage", None)
        self.register_buffer("spiking", None)

    def reset_state(self, batch_size, num_steps, device):
        # State dimensions
        self.batch_size = batch_size
        self.num_steps  = num_steps

        # Initialise state variables (NOTE: +1 time step to store initial state)
        self.current = torch.zeros((batch_size, num_steps + 1, self.out_features), device=device)
        self.voltage = torch.zeros((batch_size, num_steps + 1, self.out_features), device=device)
        self.spiking = torch.zeros((batch_size, num_steps + 1, self.out_features), device=device)

    def forward(self, input):
        # Reset layer state
        self.reset_state(input.shape[0], input.shape[1], input.device)

        # Simulate the discretised LIF dynamics
        for t in range(self.num_steps):
            # NOTE: Detach reset term to prevent gradients flowing through it, also see
            #       https://github.com/fzenke/spytorch/issues/2#issuecomment-477685845
            self.voltage[:, t + 1] = (self.beta_mem * self.voltage[:, t] + (1 - self.beta_mem) * self.current[:, t]) * (1.0 - self.spiking[:, t].detach())
            self.current[:, t + 1] = self.beta_syn * self.current[:, t] + torch.mm(input[:, t], self.weight.t())
            self.spiking[:, t + 1] = self.spike_fun(self.voltage[:, t + 1])

        return self.spiking, self.voltage, self.current

    @staticmethod
    def spike_fun(x):
        return HeavisideFastSigmoid.apply(x - 1.0)
