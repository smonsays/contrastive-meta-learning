"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import jax
import jax.tree_util as jtu
import optax

import data
import utils


class MetaLearner:
    """
    Wraps a meta model and enables meta-learning on it.
    """
    # Should be able to specify MAML, IFT and CML derived from this class
    def __init__(
        self, meta_model, optim_fn_inner, batch_size, steps_inner
    ):
        self.meta_model = meta_model
        self.optim_fn_inner = optim_fn_inner
        self.batch_size = batch_size
        self.steps_inner = steps_inner

    def adapt(self, rng, params, hparams, inputs, targets):

        def inner_loss(batch, params, hparams):
            pred = self.meta_model(params, hparams, batch.x)

            return self.meta_model.loss_fn_inner(pred, batch.y, params, hparams)

        def inner_step(state, batch):
            params, hparams, optim = state

            loss, grads = jax.value_and_grad(inner_loss, 1)(batch, params, hparams)
            params_update, optim = self.optim_fn_inner.update(grads, optim)
            params = optax.apply_updates(params, params_update)

            metrics = {
                "loss_inner": loss,
                "gradnorm_inner": optax.global_norm(grads)
            }

            return [params, hparams, optim], metrics

        optim_init = self.optim_fn_inner.init(params)

        batch_loader = data.batch_generator(
            rng, inputs, targets, self.batch_size, self.steps_inner
        )

        carry, metrics = jax.lax.scan(inner_step, [params, hparams, optim_init], batch_loader)
        params, hparams, optim = carry

        return params, metrics

    def eval(self, rng, hparams, extended_dataset):
        rng_adapt, rng_reset = jax.random.split(rng)
        params = self.meta_model.reset_params(rng_reset, hparams, extended_dataset.x_train)
        params, metrics_inner = self.adapt(
            rng_adapt, params, hparams, extended_dataset.x_train, extended_dataset.y_train
        )
        pred = self.meta_model(params, hparams, extended_dataset.x_test)
        loss = self.meta_model.loss_fn_outer(pred, extended_dataset.y_test, params, hparams)

        return loss, metrics_inner

    def grad(self, rng, hparams, extended_dataset):
        raise NotImplementedError


class MAML(MetaLearner):

    def grad(self, rng, hparams, extended_dataset):
        eval_value_and_grad = jax.value_and_grad(self.eval, argnums=1, has_aux=True)
        (loss, metrics_inner), grads = eval_value_and_grad(rng, hparams, extended_dataset)

        metrics = {
            "loss_outer": loss,
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_inner
        }

        return grads, metrics


class EquilibriumPropagation(MetaLearner):
    def __init__(
        self,
        meta_model,
        optim_fn_inner,
        optim_fn_nudged,
        batch_size,
        steps_inner,
        steps_nudged,
        beta,
    ):
        super().__init__(
            meta_model, optim_fn_inner, batch_size, steps_inner
        )
        self.optim_fn_nudged = optim_fn_nudged
        self.steps_nudged = steps_nudged
        self.beta = beta

    def augmented_loss(self, batch_train, batch_test, params, hparams, beta):
        pred_train = self.meta_model(params, hparams, batch_train.x)
        energy = self.meta_model.loss_fn_inner(pred_train, batch_train.y, params, hparams)

        pred_test = self.meta_model(params, hparams, batch_test.x)
        cost = self.meta_model.loss_fn_outer(pred_test, batch_test.y, params, hparams)

        return energy + beta * cost, {"cost": cost, "energy": energy}

    def adapt_augmented(self, rng, params, hparams, beta, extended_dataset):
        augmented_loss_grad = jax.value_and_grad(self.augmented_loss, 2, has_aux=True)

        def augmented_step(state, batch_train_test):
            params, hparams, optim = state
            batch_train, batch_test = batch_train_test

            (loss, metrics_nudged), grads = augmented_loss_grad(
                batch_train, batch_test, params, hparams, beta
            )
            params_update, optim = self.optim_fn_nudged.update(grads, optim)
            params = optax.apply_updates(params, params_update)

            metrics = {
                "loss_nudged": loss,
                "gradnorm_nudged": optax.global_norm(grads),
                **metrics_nudged,
            }

            return [params, hparams, optim], metrics

        optim_init = self.optim_fn_nudged.init(params)

        train_loader = data.batch_generator(
            rng, extended_dataset.x_train, extended_dataset.y_train,
            self.batch_size, self.steps_nudged
        )
        # NOTE: When batch_size > len(extended_dataset.x_test), this will dupilicate samples
        #       within a single batch. As long as we average the loss, this should not be an issue
        test_loader = data.batch_generator(
            rng, extended_dataset.x_test, extended_dataset.y_test,
            self.batch_size, self.steps_nudged
        )

        carry, metrics = jax.lax.scan(
            augmented_step, [params, hparams, optim_init], (train_loader, test_loader)
        )
        params, hparams, optim = carry

        return params, metrics

    def grad(self, rng, hparams, extended_dataset):
        rng_free, rng_nudged, rng_reset, rng_train, rng_test = jax.random.split(rng, 5)

        # Free phase
        params = self.meta_model.reset_params(rng_reset, hparams, extended_dataset.x_train)
        params_free, metrics_free = self.adapt(
            rng_free, params, hparams, extended_dataset.x_train, extended_dataset.y_train
        )

        # Nudged phase
        params_nudged, metrics_nudged = self.adapt_augmented(
            rng_nudged, params_free, hparams, self.beta, extended_dataset
        )

        # Evaluate partial derivatives of augmented loss wrt hparams on single, random batch
        # TODO: Evaluate augmented loss on single batch or full data?
        # batch_train = data.get_batch(
        #     rng_train, extended_dataset.x_train, extended_dataset.y_train, self.batch_size
        # )
        # batch_test = data.get_batch(
        #     rng_test, extended_dataset.x_test, extended_dataset.y_test, self.batch_size
        # )
        batch_train = data.Dataset(extended_dataset.x_train, extended_dataset.y_train)
        batch_test = data.Dataset(extended_dataset.x_test, extended_dataset.y_test)

        augmented_loss_hgrad = jax.grad(self.augmented_loss, 3, has_aux=True)
        grads_free, _ = augmented_loss_hgrad(
            batch_train, batch_test, params_free, hparams, beta=0.0
        )
        grads_nudged, _ = augmented_loss_hgrad(
            batch_train, batch_test, params_nudged, hparams, beta=self.beta
        )

        # Compute the meta-gradient
        def ep_first_order(g_free, g_nudged):
            return (1.0 / self.beta) * (g_nudged - g_free)

        grads = jtu.tree_map(ep_first_order, grads_free, grads_nudged)

        metrics = {
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_free,
            **metrics_nudged
        }

        return grads, metrics


class SymmetricEquilibriumPropagation(EquilibriumPropagation):

    def grad(self, rng, hparams, extended_dataset):
        rng_free, rng_nudged_pos, rng_nudged_neg, rng_reset, rng_train, rng_test = jax.random.split(rng, 6)

        # Free phase
        params = self.meta_model.reset_params(rng_reset, hparams, extended_dataset.x_train)
        params_free, metrics_free = self.adapt(
            rng_free, params, hparams, extended_dataset.x_train, extended_dataset.y_train
        )

        # Nudged phases
        params_nudged_pos, metrics_nudged_pos = self.adapt_augmented(
            rng_nudged_pos, params_free, hparams, self.beta, extended_dataset
        )
        params_nudged_neg, metrics_nudged_neg = self.adapt_augmented(
            rng_nudged_neg, params_free, hparams, -self.beta, extended_dataset
        )

        # Evaluate partial derivatives of augmented loss wrt hparams on single, random batch
        # TODO: Evaluate augmented loss on single batch or full data?
        # batch_train = data.get_batch(
        #     rng_train, extended_dataset.x_train, extended_dataset.y_train, self.batch_size
        # )
        # batch_test = data.get_batch(
        #     rng_test, extended_dataset.x_test, extended_dataset.y_test, self.batch_size
        # )
        batch_train = data.Dataset(extended_dataset.x_train, extended_dataset.y_train)
        batch_test = data.Dataset(extended_dataset.x_test, extended_dataset.y_test)

        augmented_loss_hgrad = jax.grad(self.augmented_loss, 3, has_aux=True)
        grads_nudged_pos, _ = augmented_loss_hgrad(
            batch_train, batch_test, params_nudged_pos, hparams, beta=self.beta
        )
        grads_nudged_neg, _ = augmented_loss_hgrad(
            batch_train, batch_test, params_nudged_neg, hparams, beta=-self.beta
        )

        # Compute the meta-gradient
        def ep_symmetric(g_nudged_pos, g_nudged_neg):
            return (1 / (2.0 * self.beta)) * (g_nudged_pos - g_nudged_neg)

        grads = jtu.tree_map(ep_symmetric, grads_nudged_pos, grads_nudged_neg)

        metrics = {
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_free,
            **metrics_nudged_pos,
            **utils.append_keys(metrics_nudged_neg, "neg")
        }

        return grads, metrics
