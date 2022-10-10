"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


class BayesianLinearRegression(nn.Module):
    """
    Bayesian linear regression with a posterior parametrised as
    pi(beta, sigma^2) = Normal(beta|sigma^2) * InvGamma(sigma^2).
    """
    feature_dim: int
    a0: float
    b0: float
    lambda_prior: float
    intercept: bool

    def setup(self):
        def diag_init(scale):
            def init_fn(rng, shape):
                return scale * jnp.eye(shape[0])
            return init_fn

        def const_init(value):
            # TODO: Can be replaced with built-in nn.initializers.constant after updating jax
            def init_fn(rng, shape):
                return jnp.full(shape, value)
            return init_fn

        # Setup inverse gamma parameters
        self.a = self.param('a', const_init(self.a0), ())
        self.b = self.param('b', const_init(self.b0), ())

        # Setup Gaussian parameters
        self.mu = self.param('mu', nn.initializers.zeros, (self.feature_dim + self.intercept, ))
        self.cov = self.param(
            'cov',
            diag_init(1.0 / self.lambda_prior),
            (self.feature_dim + self.intercept, )
        )
        self.precision = self.param(
            'precision',
            diag_init(self.lambda_prior),
            (self.feature_dim + self.intercept, )
        )

    def __call__(self, rng, x):
        """
        Sample a single set of parameters from the posterior and return prediction.
        """
        rng_gamma, rng_gaussian = jax.random.split(rng, 2)

        # Sample sigma^2 and beta from posterior
        sigma2_s = self.b / jax.random.gamma(rng_gamma, self.a)
        beta_s = jax.random.multivariate_normal(rng_gaussian, self.mu, sigma2_s * self.cov)

        # NOTE: This could fail if covariance is not positive definite
        # assert all(jnp.linalg.eigvalsh(self.cov))
        # pos_definite = jnp.all(jnp.linalg.eigvalsh(self.cov))

        # Sample from unit Gaussian as backup
        # beta_s_fallback = jax.random.multivariate_normal(
        #     rng_gaussian, jnp.zeros((self.feature_dim + self.intercept)), jnp.eye(self.feature_dim + self.intercept)
        # )
        # beta_s = pos_definite * beta_s + (1.0 - pos_definite) * beta_s_fallback

        if self.intercept:
            return jnp.dot(beta_s[:-1], x.T) + beta_s[-1]
        else:
            return jnp.dot(beta_s, x.T)

    def fit(self, x, y):
        """
        Fit params to data x,y using non-sequential update (i.e. everything is recomputed)
        """
        mask = jnp.all(x != 0, axis=1)
        num_data = jnp.sum(mask)  # Instead of x.shape[0] to for allow masking

        if self.intercept:
            x = jnp.column_stack((x, mask))

        precision = jnp.dot(x.T, x) + self.lambda_prior * jnp.eye(self.feature_dim + self.intercept)
        cov = jnp.linalg.inv(precision)
        mu = jnp.dot(cov, jnp.dot(x.T, y))

        # Inverse Gamma posterior update
        a = self.a0 + num_data / 2.0
        b = self.b0 + 0.5 * (jnp.dot(y.T, y) - jnp.dot(mu.T, jnp.dot(precision, mu)))

        return FrozenDict({
            "params": {
                "a": a,
                "b": b,
                "mu": mu,
                "cov": cov,
                "precision": precision,
            }
        })


class RidgeRegression(nn.Module):
    feature_dim: int
    l2_reg: float
    intercept: bool

    def setup(self):
        self.weight = self.param(
            'weight', nn.initializers.zeros, (self.feature_dim + self.intercept, )
        )

    def __call__(self, x):
        if self.intercept:
            return jnp.dot(self.weight[:-1], x.T) + self.weight[-1]
        else:
            return jnp.dot(self.weight, x.T)

    def fit(self, x, y):
        """
        Minimize ridge loss 0.5 * (jnp.mean((jnp.dot(x, w) - y) ** 2) + l2_reg * jnp.sum(w ** 2))
        using the conjugate gradient method.
        """
        if self.intercept:
            mask = jnp.all(x != 0, axis=1)
            x = jnp.column_stack((x, mask))

        def matvec(u):
            return jnp.dot(x.T, jnp.dot(x, u)) + self.l2_reg * u

        weight = jax.scipy.sparse.linalg.cg(matvec, jnp.dot(x.T, y), x0=self.weight, maxiter=10)[0]

        return FrozenDict({
            "params": {
                "weight": weight,
            }
        })
