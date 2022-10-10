"""
Copyright (c) Simon Schug, adapted from ***anonymized***.
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math

import torch

import energy

from .layers.surrogate import HeavisideFastSigmoid


class RecurrentSpikingNetwork(torch.nn.Module):

    def __init__(self, dimensions, tau_hidden, tau_output, step_size, lr_modulate=[1.0, 1.0, 1.0],
                 threshold=0.1, gamma=0.3, w_init_gain=(0.1, 0.01, 0.1), feedback_align=False):
        super().__init__()

        assert len(dimensions) == 3

        self.num_in, self.num_rec, self.num_out = dimensions
        self.step_size = step_size
        self.lr_modulate = lr_modulate
        self.threshold = threshold
        self.gamma = gamma
        self.gain = w_init_gain
        self.feedback_align = feedback_align

        # Set time-constants
        self.alpha = math.exp(-step_size / tau_hidden)
        self.kappa = math.exp(-step_size / tau_output)

        # Parameters
        self.weight_in  = torch.nn.Parameter(torch.empty(self.num_rec, self.num_in))
        self.weight_rec = torch.nn.Parameter(torch.empty(self.num_rec, self.num_rec))
        self.weight_out = torch.nn.Parameter(torch.empty(self.num_out, self.num_rec))

        if self.feedback_align:
            self.register_buffer("weight_fa", torch.empty(self.num_out, self.num_rec))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight_in)
        self.weight_in.data = self.gain[0] * self.weight_in.data

        torch.nn.init.kaiming_normal_(self.weight_rec)
        self.weight_rec.data = self.gain[1] * self.weight_rec.data

        torch.nn.init.kaiming_normal_(self.weight_out)
        self.weight_out.data = self.gain[2] * self.weight_out.data

        if self.feedback_align:
            self.weight_fa = torch.nn.init.kaiming_normal_(self.weight_fa)
            self.weight_fa.data = self.gain[2] * self.weight_fa.data

    def forward(self, input):
        # Reset network state
        batch_size, num_steps = input.shape[0], input.shape[1]
        v     = torch.zeros(batch_size, self.num_rec, device=input.device)
        v_out = torch.zeros(batch_size, self.num_out, device=input.device)
        z     = torch.zeros(batch_size, self.num_rec, device=input.device)

        # Collect the time-series of state variables in lists
        vs, v_outs, zs = [v], [v_out], [z]

        for t in range(num_steps - 1):
            # Unfold the dynamics
            v = (self.alpha * v + torch.mm(z, self.weight_rec.t()) + torch.mm(input[:, t], self.weight_in.t())) - (z * self.threshold).detach()
            z  = HeavisideFastSigmoid.apply(v - self.threshold)
            v_out = self.kappa * v_out + torch.mm(z, self.weight_out.t())

            vs.append(v), v_outs.append(v_out), zs.append(z)

        # Output average potential of each readout unit over time
        output = torch.mean(torch.stack(v_outs, dim=1), dim=1)

        return output, {"spiking": torch.stack(zs, dim=1), "voltage": torch.stack(vs, dim=1), "v_out": torch.stack(v_outs, dim=1)}

    def custom_grad(self, input, target, state, loss_function, hyperparams):
        """
        Custom grad implementation using eprop.
        """
        # Extract the state for which to compute approximate gradients
        batch_size, num_steps = input.shape[0], input.shape[1]
        z, v, v_out = state["spiking"], state["voltage"], state["v_out"]

        # Surrogate derivative
        spike_grad = self.heavyside_surrogate_grad(v, self.threshold, self.gamma)

        # Input eligibility vector
        alpha_conv = torch.tensor([
            self.alpha ** (num_steps - i - 1) for i in range(num_steps)
        ], device=input.device, dtype=torch.float).view(1, 1, -1)

        trace_in = torch.nn.functional.conv1d(
            input.permute(0, 2, 1), alpha_conv.expand(self.num_in, -1, -1),
            padding=num_steps,
            groups=self.num_in
        )[:, :, 1:num_steps + 1].unsqueeze(1).expand(-1, self.num_rec, -1, -1)
        trace_in = torch.einsum('btr, brit -> brit', spike_grad, trace_in)

        # Recurrent eligibility vector
        trace_rec = torch.nn.functional.conv1d(
            z.permute(0, 2, 1), alpha_conv.expand(self.num_rec, -1, -1),
            padding=num_steps, groups=self.num_rec
        )[:, :, :num_steps].unsqueeze(1).expand(-1, self.num_rec, -1, -1)
        trace_rec = torch.einsum('btr, brit -> brit', spike_grad, trace_rec)

        # Output eligibility vector
        kappa_conv = torch.tensor([
            self.kappa ** (num_steps - i - 1) for i in range(num_steps)
        ], device=input.device, dtype=torch.float).view(1, 1, -1)

        trace_out = torch.nn.functional.conv1d(
            z.permute(0, 2, 1), kappa_conv.expand(self.num_rec, -1, -1),
            padding=num_steps, groups=self.num_rec
        )[:, :, 1:num_steps + 1]

        # Low-pass filter eligibility traces for MSE loss
        trace_in_low = torch.nn.functional.conv1d(
            trace_in.reshape(batch_size, self.num_in * self.num_rec, num_steps),
            kappa_conv.expand(self.num_in * self.num_rec, -1, -1),
            padding=num_steps, groups=self.num_in * self.num_rec
        )[:, :, 1:num_steps + 1].reshape(batch_size, self.num_rec, self.num_in, num_steps)

        trace_rec_low = torch.nn.functional.conv1d(
            trace_rec.reshape(batch_size, self.num_rec * self.num_rec, num_steps),
            kappa_conv.expand(self.num_rec * self.num_rec, -1, -1),
            padding=num_steps, groups=self.num_rec * self.num_rec
        )[:, :, 1:num_steps + 1].reshape(batch_size, self.num_rec, self.num_rec, num_steps)

        # Semi-automatically determine which losses should be active for a number of hard-coded loss functions
        # NOTE: This is certainly not bullet-proof and only works for very specific loss functions
        supported_losses = (
            energy.MeanSquaredError, energy.MeanFiringRate,
            energy.ElasticRegularizer, energy.EnergySum, torch.nn.ModuleList
        )
        for m in loss_function.modules():
            assert isinstance(m, supported_losses), "Loss function {} not implemented with eprop".format(m)

        # Initialise tensors to store the gradients
        weight_in_grad  = torch.zeros_like(self.weight_in)
        weight_rec_grad  = torch.zeros_like(self.weight_rec)
        weight_out_grad  = torch.zeros_like(self.weight_out)

        # Learning signal: mean squared error
        if any([isinstance(m, energy.MeanSquaredError) for m in loss_function.modules()]):
            # Expand the target signal for every time step
            target_dense = target.unsqueeze(dim=1).expand(-1, num_steps, -1)

            # Backpropagate the error either exactly or using feedback alignment
            if self.feedback_align:
                error_mse = torch.einsum('bto, or -> brt', v_out - target_dense, self.weight_fa)
            else:
                error_mse = torch.einsum('bto, or -> brt', v_out - target_dense, self.weight_out)

            weight_in_grad  += torch.sum(error_mse.unsqueeze(2).expand(-1, -1, self.num_in, -1) * trace_in_low, dim=(0, 3))
            weight_rec_grad += torch.sum(error_mse.unsqueeze(2).expand(-1, -1, self.num_rec, -1) * trace_rec_low, dim=(0, 3))
            weight_out_grad += torch.einsum('bto, brt -> or', v_out - target_dense, trace_out)

        # Learning signal: mean activity regularizer
        if any([isinstance(m, energy.MeanFiringRate) for m in loss_function.modules()]):
            # Extract the module hyperparameters
            reg_module = [m for m in loss_function.modules() if isinstance(m, energy.MeanFiringRate)][0]
            reg_strength, target_rate = reg_module.strength, reg_module.target

            error_reg_activity = reg_strength * (torch.mean(z, dim=1) - target_rate).unsqueeze(-1).expand(-1, -1, num_steps)

            weight_in_grad  += torch.sum(error_reg_activity.unsqueeze(2).expand(-1, -1, self.num_in, -1) * trace_in, dim=(0, 3))
            weight_rec_grad += torch.sum(error_reg_activity.unsqueeze(2).expand(-1, -1, self.num_rec, -1) * trace_rec, dim=(0, 3))

        # Learning signal: elastic regularizer
        if any([isinstance(m, energy.ElasticRegularizer) for m in loss_function.modules()]):
            # Extract the module hyperparameters
            reg_module = [m for m in loss_function.modules() if isinstance(m, energy.ElasticRegularizer)][0]
            reg_strength = reg_module.l2_strength

            error_reg_imaml_in = reg_strength * (self.weight_in - hyperparams["omega"][0])
            error_reg_imaml_rec = reg_strength * (self.weight_rec - hyperparams["omega"][1])
            error_reg_imaml_out = reg_strength * (self.weight_out - hyperparams["omega"][2])

            weight_in_grad += error_reg_imaml_in
            weight_rec_grad += error_reg_imaml_rec
            weight_out_grad += error_reg_imaml_out

        # Layer-wise modulation of learning rates
        weight_in_grad  *= self.lr_modulate[0]
        weight_rec_grad *= self.lr_modulate[1]
        weight_out_grad *= self.lr_modulate[2]

        return [weight_in_grad, weight_rec_grad, weight_out_grad]

    @staticmethod
    def heavyside_surrogate_grad(v, threshold, gamma):
        return gamma * torch.max(torch.zeros_like(v), 1 - torch.abs((v - threshold) / threshold))

    def extra_repr(self):
        return "{} -> {} -> {}".format(self.num_in, self.num_rec, self.num_out)
