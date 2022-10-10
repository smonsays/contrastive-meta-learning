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
from .utils import find_by_subkey


class CrossEntropy(EnergyFunction):
    def forward(self, output, target, meta_model, state):
        return torch.nn.functional.cross_entropy(output, target, reduction="mean")


class EvidenceLowerBound(EnergyFunction):
    def __init__(self, num_batches, beta=None):
        super().__init__()
        self.num_batches = num_batches
        self.beta = beta

    @staticmethod
    def _kl_divergence_gaussian(mu_0, mu_1, log_sigma_0, log_sigma_1):
        """
        Analytical computation of KL-divergence between two Normal distribtuions (see Graves 2011).

        Args:
            mu_0: mean of normal distribution (posterior).
            mu_1: mean of normal distribution (prior).
            log_sigma_0: logarithm of standard deviation of normal distribution (posterior).
            log_sigma_1: logarithm of standard deviation of normal distribution (prior).
        """
        return torch.sum(
            log_sigma_1 - log_sigma_0 + (torch.exp(log_sigma_0)**2 + (mu_0 - mu_1)**2) / (2 * torch.exp(log_sigma_1)**2) - 0.5
        )

    def forward(self, output, target, meta_model, state):
        nll_loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
        kld_loss = torch.zeros(len(meta_model.base_learner.layers), device=output.device)

        for l, layer in enumerate(meta_model.base_learner.layers):
            # Extract the hyperparameters for this layer from the flat hyperparameter dictionary
            # NOTE: This is a bit ugly but we need to match hyperparams.parameters() and model.parameters() elsewhere
            prior_bias_mu = find_by_subkey(meta_model.meta_learner["prior"], [str(l), "bias", "mu"])
            prior_bias_log_sigma = find_by_subkey(meta_model.meta_learner["prior"], [str(l), "bias", "log_sigma"])
            prior_weight_mu = find_by_subkey(meta_model.meta_learner["prior"], [str(l), "weight", "mu"])
            prior_weight_log_sigma = find_by_subkey(meta_model.meta_learner["prior"], [str(l), "weight", "log_sigma"])

            # KLD on biases
            kld_loss[l] += self._kl_divergence_gaussian(
                prior_bias_mu, layer.bias_mu, prior_bias_log_sigma, layer.bias_log_sigma
            )
            # KLD on weights
            kld_loss[l] += self._kl_divergence_gaussian(
                prior_weight_mu, layer.weight_mu, prior_weight_log_sigma, layer.weight_log_sigma
            )

        if self.beta is None:
            elbo = nll_loss + torch.sum(meta_model.meta_learner['prior_strength'][0] * kld_loss) / self.num_batches
        else:
            elbo = nll_loss + self.beta * torch.sum(kld_loss) / self.num_batches

        return elbo / len(output)


class MeanSquaredError(EnergyFunction):
    def forward(self, output, target, meta_model, state):
        return torch.nn.functional.mse_loss(output, target, reduction="mean")


class SmoothL1Loss(EnergyFunction):
    def forward(self, output, target, meta_model, state):
        return torch.nn.functional.smooth_l1_loss(output, target, reduction="mean")
