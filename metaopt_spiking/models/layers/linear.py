"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import itertools
import math

import torch

from .surrogate import FeedbackAlignLinearFun


class BayesianLinear(torch.nn.Module):
    """
    Bayesian (Gaussian) linear layer.
    """

    def __init__(self, in_features, out_features, stdv_init=None, radial=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.stdv_init = stdv_init
        self.radial = radial

        # Weight parameters
        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        # Bias parameters
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = torch.nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_eps', None)

        # Initialize parameters
        self.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, radial={}'.format(
            self.in_features, self.out_features, self.radial
        )

    def forward(self, input):
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps

        return torch.nn.functional.linear(input, weight, bias)

    def reset_parameters(self):
        """
        Initialize all tunable parameters of the layer
        """
        # Kaiming normal initialisation for weights mean
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        fan_in_std = 1 / math.sqrt(fan_in)
        torch.nn.init.normal_(self.weight_mu, mean=0.0, std=1.0 / math.sqrt(fan_in))

        # Kaiming uniform initialisation for bias mean
        torch.nn.init.uniform_(self.bias_mu, -fan_in_std, fan_in_std)

        # Initialise standard deviations with constant values
        stdv_transform = math.log(math.exp(self.stdv_init) - 1)
        self.weight_log_sigma.data.fill_(stdv_transform)
        self.bias_log_sigma.data.fill_(stdv_transform)

        # Sample parameters
        self.sample()

    @staticmethod
    def _radial_normalize(eps):
        radial_distance = torch.randn((1), device=eps.device)
        eps = eps / torch.norm(eps, p=2) * radial_distance

        return eps

    def sample(self):
        if self.radial:
            self.weight_eps = self._radial_normalize(torch.randn_like(self.weight_log_sigma))
            self.bias_eps = self._radial_normalize(torch.randn_like(self.bias_log_sigma))
        else:
            self.weight_eps = torch.randn_like(self.weight_log_sigma)
            self.bias_eps = torch.randn_like(self.bias_log_sigma)


class FeedbackAlignLinear(torch.nn.Module):
    """
    Linear module with Feedback Alignment for the backward pass.
    """
    def __init__(self, in_features, out_features, bias, feedback_init=0.1):
        super(FeedbackAlignLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.feedback_init = feedback_init

        self.weight = torch.nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_fa', torch.empty(out_features, in_features, requires_grad=False))

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        fan_out_std = 1.0 / math.sqrt(self.weight_fa.size(0))
        torch.nn.init.normal_(self.weight_fa, std=self.feedback_init * fan_out_std)

    def forward(self, input):
        return FeedbackAlignLinearFun.apply(input, self.weight, self.bias, self.weight_fa)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MetaGainLinear(torch.nn.Linear):
    def forward(self, input, params):
        return torch.nn.functional.linear(input, self.weight, self.bias) * torch.sigmoid(params['gain'])


class SlicedLinear(torch.nn.Linear):
    """
    Linear layer that slices its output.

    NOTE: This is only faster than sequentially calling Linear when run using cuda.
    """
    def __init__(self, in_features, out_features_list, bias=True):
        super().__init__(in_features, sum(out_features_list), bias)
        # This is a python 3.8 feature, not all machines are ready for it yet
        # total = 0
        # self.slice_indeces = [0] + [total := total + dim for dim in out_features_list]
        self.slice_indeces = [0] + list(itertools.accumulate(out_features_list))

    def forward(self, input):
        out = super().forward(input)

        out_sliced = [
            out[:, start:end]
            for start, end in zip(self.slice_indeces[:-1], self.slice_indeces[1:])
        ]

        return out_sliced
