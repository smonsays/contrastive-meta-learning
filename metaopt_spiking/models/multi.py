"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from .container import Parallel


class MultiheadNetwork(torch.nn.Module):
    def __init__(self, features, output_dim, num_outputs, nonlinearity):
        super().__init__()

        self.features = features
        self.input_dim = features.input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.nonlinearity = nonlinearity

        self.classifiers = Parallel(
            torch.nn.Linear(self.features.output_dim, output_dim)
            for _ in range(num_outputs)
        )

    def forward(self, input):
        # Split the input into network input and task id
        x, task = torch.split(input, [input.shape[-1] - 1, 1], dim=-1)

        x, state = self.features(x)
        state["activation"].append(x := self.nonlinearity(x))

        # Compute outputs over all classifiers and then select by task which one to use
        outputs = self.classifiers(x)
        output = outputs[task.squeeze().long(), range(len(task)), :]

        return output, state

    def reset_parameters(self):
        self.features.reset_parameters()
        self.classifiers.reset_parameters()


class SingleheadNetwork(torch.nn.Module):
    def __init__(self, features, output_dim, nonlinearity):
        super().__init__()

        self.features = features
        self.input_dim = features.input_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity

        self.classifier = torch.nn.Linear(self.features.output_dim, output_dim)

    def forward(self, input):
        x, state = self.features(input)
        state["activation"].append(x := self.nonlinearity(x))

        return self.classifier(x), state

    def reset_parameters(self):
        self.features.reset_parameters()
        self.classifier.reset_parameters()
