"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
import copy

import torch


class MetaModule(abc.ABC, torch.nn.Module):
    """
    Abstract interface for meta modules (e.g. Hyperopt, Hypernet, ModularNet).
    TODO: Hypernet is not yet defined within this interface but might require
          the forward method to take an optional params argument (see models.hnet)
    """
    def __init__(self, base_learner, meta_learner):
        super().__init__()
        self.base_learner = base_learner
        self.meta_learner = meta_learner

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        pass

    def base_parameters(self):
        return self.base_learner.parameters()

    def meta_parameters(self):
        return self.meta_learner.parameters()


class Hyperopt(MetaModule):
    def __init__(self, model, hyperparams, inner_init, theta_key=None, nonnegative_keys={}):
        super().__init__(model, hyperparams)

        assert inner_init in ["fixed_seed", "from_theta", "reset"]

        self.inner_init = inner_init
        self.theta_key = theta_key
        self.nonnegative_keys = nonnegative_keys

        if self.inner_init == "fixed_seed":
            self.model_init_state = copy.deepcopy(self.base_learner.state_dict())

    def forward(self, input):
        return self.base_learner(input)

    def reset_parameters(self):
        if self.inner_init == "fixed_seed":
            # Initialise model parameters to fixed parameter initialisation
            self.base_learner.load_state_dict(self.model_init_state)

        elif self.inner_init == "from_theta":
            # Initialise model parameters to corresponding hyperparameters
            with torch.no_grad():
                for p, hp in zip(self.base_parameters(), self.meta_learner[self.theta_key].parameters()):
                    p.copy_(hp)

        if self.inner_init == "reset":
            # Initialise model parameters randomly
            self.base_learner.reset_parameters()

    def enforce_nonnegativity(self):
        """
        Enforce non-negativity through projected GD for selected hyperparameters.
        """
        for key in set(self.meta_learner.keys()).intersection(self.nonnegative_keys):
            for hp in self.meta_learner[key].parameters():
                with torch.no_grad():
                    hp.relu_()


class ModularNet(MetaModule):
    def __init__(self, classifier, features):
        super().__init__(classifier, features)

    def forward(self, input):
        x, state_features = self.meta_learner(input)
        x, state_classifier = self.base_learner(x)

        state = {
            "activation_features": state_features["activation"],
            "activation_classifier": state_classifier["activation"],
        }

        return x, state

    def reset_parameters(self):
        return self.base_learner.reset_parameters()
