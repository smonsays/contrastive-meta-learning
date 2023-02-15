# A contrastive rule for meta-learning

Official implementation of the paper [A contrastive rule for meta-learning](https://arxiv.org/abs/2104.01677) published at NeurIPS 2022.

## Usage

- [`metaopt_spiking/`](https://github.com/smonsays/contrastive-meta-learning/blob/main/metaopt_spiking) implements the meta-optimization experiments of section 5.2 and the recurrent spiking network experiments of section 5.4
- [`fewshot/`](https://github.com/smonsays/contrastive-meta-learning/blob/main/fewshot) implements the visual few-shot experiments of section 5.3
- [`bandit/`](https://github.com/smonsays/contrastive-meta-learning/blob/main/bandit) implements the reward-based learning experiment of section 5.5

## Dependencies

The meta-optimization (section 5.2), visual few-shot learning (section 5.3) and recurrent spiking network (section 5.4) experiments are implemented using [pytorch](https://github.com/pytorch/pytorch), the reward-based learning experiment (section 5.5) is implemented using [jax](https://github.com/google/jax).

For specific package dependencies see the respective subfolder's `requirements.txt` files.
