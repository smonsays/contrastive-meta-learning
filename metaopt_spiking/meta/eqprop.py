
"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
class EquilibriumPropagation:
    def __init__(self, beta, variant="standard"):
        self.beta = beta
        self.variant = variant

    def hypergrad(self, grad_hps_free, grad_hps_nudged_dict):
        if self.variant == "standard":
            return self._standard(grad_hps_free, grad_hps_nudged_dict)

        elif self.variant == "symmetric":
            return self._symmetric(grad_hps_nudged_dict)

        elif self.variant == "second_order":
            return self._second_order(grad_hps_free, grad_hps_nudged_dict)
        else:
            raise ValueError("EP variant \"{}\" undefined".format(self.variant))

    def _standard(self, grad_hps_free, grad_hps_nudged_dict):
        """
        Standard first-order forward difference to estimate hyperparameter gradient.
        """
        return [
            (1.0 / self.beta) * (grad_nudged - grad_free)
            for grad_free, grad_nudged in zip(grad_hps_free, grad_hps_nudged_dict[self.beta])
        ]

    def _symmetric(self, grad_hps_nudged_dict):
        """
        Central difference to estimate hyperparameter gradient.
        """
        return [
            (1 / (2.0 * self.beta)) * (grad_nudged_plus - grad_nudged_minus)
            for grad_nudged_plus, grad_nudged_minus in zip(
                grad_hps_nudged_dict[self.beta], grad_hps_nudged_dict[-self.beta]
            )
        ]

    def _second_order(self, grad_hps_free, grad_hps_nudged_dict):
        """
        Second-order forward difference to estimate hyperparameter gradient.
        """
        return [
            (1.0 / self.beta) * (-1.5 * grad_free + 2.0 * grad_nudged - 0.5 * grad_nudged2)
            for grad_free, grad_nudged, grad_nudged2 in zip(
                grad_hps_free, grad_hps_nudged_dict[self.beta],
                grad_hps_nudged_dict[2 * self.beta],
            )
        ]
