# A contrastive rule for meta-learning

This repository implements the meta-reinforcement learning experiments on the wheel bandit task.

## Experiments

To run the experiments reported in the paper you may execute the follwing commands.

```
python ray_metabandit.py --method cml --meta_model imaml
python ray_metabandit.py --method cml --meta_model gain_mod
python ray_metabandit.py --method maml --meta_model learned_init
```

## Dependencies

Dependencies are defined in `requirements.txt` and can be installed via `pip install -r requirements.txt`