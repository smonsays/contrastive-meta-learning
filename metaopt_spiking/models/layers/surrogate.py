"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch


class FeedbackAlignLinearFun(torch.autograd.Function):
    """
    See https://pytorch.org/docs/stable/notes/extending.html (2019-07-24)
    for reference
    """
    @staticmethod
    def forward(ctx, input, weight, bias, weight_backward):
        # Store the parameters necessary to compute the backward pass
        ctx.save_for_backward(input, weight_backward, bias)

        # Perform actual linear computation using the sampled weight
        # using the same code as in the torch.nn.functional.linear (2019-07-24)
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear
        if input.dim() == 2 and bias is not None:
            # Fused op is marginally faster
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved tensors from forward pass
        input, weight, bias = ctx.saved_tensors
        # Initialize all gradients w.r.t. inputs to None
        # (trailing Nones in return statement are ignored)
        grad_input = grad_weight = grad_bias = None

        # needs_input_grad checks are optional to improve efficiency
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class ReluStraightThrough(torch.autograd.Function):
    """
    Applies max(0,x) but uses the straight-through estimator for backward.
    """
    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class HeavisideFastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient for the heavy side function using the normalized
    negative part of a fast sigmoid (see Zenke & Ganguli, 2018).
    """
    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        Shifted heaviside function of the input Tensor.
        """
        ctx.save_for_backward(input)

        return torch.heaviside(input, torch.tensor([0.0], device=input.device))

    @staticmethod
    def backward(ctx, grad_output):
        """
        Surrogate gradient as normalized negative part of a fast sigmoid.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (HeavisideFastSigmoid.scale * torch.abs((input)) + 1.0)**2

        return grad
