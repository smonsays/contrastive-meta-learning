# A contrastive rule for meta-learning

This repository implements the supervised meta-optimization and spiking few-shot regression experiments.

## Experiments

To run the experiments reported in the paper you may execute the follwing commands. Default hyperparameters are found in the `config/` folder.

## Supervised meta-optimization

### CIFAR-10
```
python run_hyperopt.py --dataset cifar10 --model lenet_l2
python run_implicit.py --dataset cifar10 --method cg --model lenet
python run_implicit.py --dataset cifar10 --method nsa --model lenet
python run_implicit.py --dataset cifar10 --method t1t2 --model lenet
python run_bptt_cifar10.py --dataset cifar10 --method tbptt --model lenet
```

### MNIST
```
python run_hyperopt.py --dataset mnist --model mlp_l2
python run_implicit.py --dataset mnist --method cg --model mlp
python run_implicit.py --dataset mnist --method nsa --model mlp
python run_implicit.py --dataset mnist --method t1t2 --model mlp
```

## Fewshot spiking regression
```
python run_fewshot.py --dataset sinusoid --model rsnn
python run_bptt_rsnn.py --dataset sinusoid --method_outer bptt --method_inner bptt
python run_bptt_rsnn.py --dataset sinusoid --method_outer bptt --method_inner eprop
python run_bptt_rsnn.py --dataset sinusoid --method_outer tbptt --method_inner eprop
```

## Dependencies

Dependencies are defined in `requirements.txt` and can be installed via `pip install -r requirements.txt`
To change the default directory for datasets, you can change the `DATAPATH` variable in `data/base.py`
