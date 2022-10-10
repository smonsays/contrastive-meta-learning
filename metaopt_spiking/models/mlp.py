"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from .base import NeuralNetwork
from .layers.linear import BayesianLinear, FeedbackAlignLinear


class BayesianMultilayerPerceptron(NeuralNetwork):
    """
    Bayesian Neural Network with various feedback mechanisms
    """
    def __init__(self, dimensions, nonlinearity, stdv_init, radial):
        super().__init__(dimensions[0], dimensions[-1])
        self.layers = torch.nn.ModuleList(
            BayesianLinear(dim1, dim2, stdv_init, radial)
            for dim1, dim2 in zip(dimensions[:-1], dimensions[1:])
        )

        self.dimensions = dimensions
        self.nonlinearity = nonlinearity

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Returns:
            y_pred: Unnormalized output (i.e. no logits or probabilities)
            state: dict containing hidden activations
        """
        x = x.view((x.size(0), self.input_dim))
        activation = []

        for l in self.layers[:-1]:
            x = l(x)
            x = self.nonlinearity(x)
            activation.append(x)

        # The output is unnormalized (i.e. no logits or probabilities)
        y_pred = self.layers[-1](x)

        return y_pred, {"activation": activation}

    def sample(self):
        for l in self.layers:
            l.sample()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()


class MultilayerPerceptron(NeuralNetwork):
    """
    Multilayer Perceptron with optional feedback alignment.

    Args:
        dimensions: List of integers defining the network architecture
        nonlinearity: Function used as nonlinearity between layers
        feedback_alignment: Boolean specifying whether to use feedback alignment
    """
    def __init__(self, dimensions, nonlinearity, feedback_alignment=False):
        super().__init__(dimensions[0], dimensions[-1])

        if feedback_alignment:
            self.layers = torch.nn.ModuleList(
                FeedbackAlignLinear(dim1, dim2, bias=True)
                for dim1, dim2 in zip(dimensions[:-1], dimensions[1:])
            )
        else:
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(dim1, dim2, bias=True)
                for dim1, dim2 in zip(dimensions[:-1], dimensions[1:])
            )
        self.dimensions = dimensions
        self.nonlinearity = nonlinearity

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Returns:
            y_pred: Unnormalized output (i.e. no logits or probabilities)
        """
        x = x.view((x.size(0), self.input_dim))
        activation = []

        for l in self.layers[:-1]:
            x = l(x)
            x = self.nonlinearity(x)
            activation.append(x)

        # The output is unnormalized (i.e. no logits or probabilities)
        y_pred = self.layers[-1](x)

        return y_pred, {"activation": activation}

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
