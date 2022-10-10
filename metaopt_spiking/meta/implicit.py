"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc

import torch


class ImplicitGradient(abc.ABC):
    """
    Approximate gradients w.r.t hyperparameters using the implicit function theorem.
    """

    @abc.abstractmethod
    def inverse_hvp(self, inner_grad, outer_grad, params):
        """
        Approximate the inverse parameter Hessian with combined application of the vector product.

        Args:
            inner_grad: gradient of the inner loss w.r.t. params (should be differentiable)
            outer_grad: gradient of the outer loss w.r.t. params
            params: inner parameters
        """
        pass

    def hypergrad(self, inner_loss, outer_loss, params, hyperparams):
        """
        Computes the (implicit) indirect gradient w.r.t. hyperparameters.

        NOTE: This assumes that the direct hyperparameter gradient
              (the gradient of the outer loss w.r.t. hyperparam) is zero.
        """
        # Compute gradient wrt parameters for inner and outer loss
        inner_grad = torch.autograd.grad(inner_loss, params, retain_graph=True, create_graph=True)
        outer_grad = torch.autograd.grad(outer_loss, params)

        # Compute the indirect gradient w.r.t. hyperparameters.
        inv_hvp = self.inverse_hvp(inner_grad, outer_grad, params)
        indirect_hypergrad = [-ge for ge in torch.autograd.grad(inner_grad, hyperparams, grad_outputs=inv_hvp)]

        return indirect_hypergrad


class ConjugateGradient(ImplicitGradient):
    def __init__(self, num_steps):
        """
        Approximate the hypergradient using the Conjugate Gradient algorithm.
        Based on https://github.com/aravindr93/imaml_dev/tree/master
        """
        self.num_steps = num_steps

    def inverse_hvp(self, inner_grad, outer_grad, params):
        """
        Approximate the inverse solving Hx = g by minimizing 1/2 x^T H x - x^T g
        with H the parameter Hessian and g the gradient of the outer loss w.r.t. the parameters.

        NOTE: In principle x could be initialised differently, but both the official iMAML [0]
              and RBP [1] implementations also initialise with zero as we do here
              [0] https://github.com/aravindr93/imaml_dev/blob/master/examples/omniglot_implicit_maml.py#L125
              [1] https://github.com/lrjconan/RBP/blob/master/utils/model_helper.py#L26
        """
        # Small helper to flatten tensor lists that span multiple contiguous subspaces ^._.^
        def flatcat(tensor_list):
            return torch.cat([tensor.contiguous().view(-1) for tensor in tensor_list])

        flat_outer_grad = flatcat(outer_grad)
        flat_inner_grad = flatcat(inner_grad)
        x = torch.zeros(flat_outer_grad.shape[0], device=flat_outer_grad.device)

        # NOTE: Technically we could compute the Hessian once here and store it but this is more
        #       cumbersome in pytorch than just repeatedly computing the full Hessian-vector product
        Ax = flatcat(
            torch.autograd.grad(flat_inner_grad, params, grad_outputs=x, retain_graph=True)
        )
        # Notation as in https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave
        r = flat_outer_grad - Ax
        p = r.clone()
        rsold = r.dot(r)

        for i in range(self.num_steps):
            Ap = flatcat(
                torch.autograd.grad(flat_inner_grad, params, grad_outputs=p, retain_graph=True)
            )
            alpha = rsold / (p.dot(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        # Reshape x to a list of tensors like params
        x_reshaped = [torch.zeros_like(p) for p in params]
        torch.nn.utils.vector_to_parameters(x, x_reshaped)

        return x_reshaped


class NeumannSeries(ImplicitGradient):
    def __init__(self, num_terms, alpha):
        """
        Approximate the hypergradient using a Neumann Series approximation (Lorraine, 2019)

        Args:
            num_terms: number of terms in the Neumann approximation
            alpha: alpha / learning rate parameter, needs to be small enough for
                   the Neumann series formulation to hold
        """
        self.num_terms = num_terms
        self.alpha = alpha

    def inverse_hvp(self, inner_grad, outer_grad, params):
        """
        Compute Neumann Series approximation of the inverse Hessian-vector product.

        Approximate the inverse as H^{-1} ≈ sum_i H^i g
        with H the parameter Hessian and g the gradient of the outer loss w.r.t. the parameters.
        """
        vec = outer_grad
        inv = outer_grad

        for i in range(self.num_terms):
            hvp = torch.autograd.grad(inner_grad, params, grad_outputs=vec, retain_graph=True)
            vec = [v - self.alpha * h for v, h in zip(vec, hvp)]
            inv = [n + v for n, v in zip(inv, vec)]

        return [self.alpha * n for n in inv]


class T1T2(ImplicitGradient):
    """
    Approximate the parameter Hessian as the identity.
    """
    def inverse_hvp(self, inner_grad, outer_grad, params):
        return outer_grad
