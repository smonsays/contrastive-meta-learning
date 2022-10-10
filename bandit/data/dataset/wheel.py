"""
Copyright 2018 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


def sample_data(
    num_contexts,
    delta,
    mean_v=[1.0, 1.0, 1.0, 1.0, 1.2],
    std_v=[0.05, 0.05, 0.05, 0.05, 0.05],
    mean_large=50.0,
    std_large=0.01,
    seed=None,
):
    """
    Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).

    Args:
      num_contexts: Number of points to sample, i.e. (context, action rewards).
      delta: Exploration parameter: high reward in one region if norm above delta.
      mean_v: Mean reward for each action if context norm is below delta.
      std_v: Gaussian reward std for each action if context norm is below delta.
      mean_large: Mean reward for optimal action if context norm is above delta.
      std_large: Reward std for optimal action if context norm is above delta.

    Returns:
      dataset: Sampled matrix with n rows: (context, action rewards).
      opt_vals: Vector of expected optimal (reward, action) for each context.
    """
    rng = np.random.default_rng(seed)

    data = []
    rewards = []
    regrets = []

    # sample uniform contexts in unit ball
    while len(data) < num_contexts:
        raw_data = rng.uniform(-1, 1, (int(num_contexts / 3), 2))

        for i in range(raw_data.shape[0]):
            if np.linalg.norm(raw_data[i, :]) <= 1:
                data.append(raw_data[i, :])

    contexts = np.stack(data)[:num_contexts, :]

    # sample rewards
    for i in range(num_contexts):
        r = [rng.normal(mean_v[j], std_v[j]) for j in range(5)]
        r_mean = np.copy(mean_v)
        if np.linalg.norm(contexts[i, :]) >= delta:
            # large reward in the right region for the context
            r_big = rng.normal(mean_large, std_large)
            if contexts[i, 0] > 0:
                if contexts[i, 1] > 0:
                    r[0] = r_big
                    r_mean[0] = mean_large
                    opt_action = 0  # upper right
                else:
                    r[1] = r_big
                    r_mean[1] = mean_large
                    opt_action = 1  # lower right
            else:
                if contexts[i, 1] > 0:
                    r[2] = r_big
                    r_mean[2] = mean_large
                    opt_action = 2  # upper left
                else:
                    r[3] = r_big
                    r_mean[3] = mean_large
                    opt_action = 3  # lower left
        else:
            opt_action = np.argmax(mean_v)

        rewards.append(r)
        regrets.append(r_mean[opt_action] - r_mean)

    rewards = np.stack(rewards)
    regrets = np.stack(regrets)

    return contexts, rewards, regrets


def sample_tasks(num_tasks, num_samples, seed=None):
    rng = np.random.default_rng(seed)

    deltas = rng.uniform(size=(num_tasks, ))
    contexts, rewards, _ = zip(*[
        sample_data(num_samples, delta=d, seed=rng)
        for d in deltas
    ])
    contexts, rewards = np.array(contexts), np.array(rewards)

    # Append one-hot actions to targets to allow masking the loss to a single arm
    actions = rng.integers(low=0, high=5, size=(num_tasks, num_samples))
    actions_one_hot = np.eye(5)[actions]
    targets = np.concatenate(
        (np.expand_dims(rewards, -2), np.expand_dims(actions_one_hot, -2)), axis=-2
    )

    return contexts, targets


def uniform_regret(delta, trials, mu_1=1.2, mu_2=1.0, mu_3=50.0):
    inner_ring = delta**2 * (0.8 * (mu_1 - mu_2))
    outer_ring = (1 - delta**2)  * (0.2 * (mu_3 - mu_1) + 0.6 * (mu_3 - mu_2))

    return trials * (inner_ring + outer_ring)
