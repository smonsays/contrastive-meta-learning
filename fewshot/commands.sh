

# miniImageNet
# 1-shot 5-ways

python3 train.py --steps_first_phase_test=100 --debug_iter=1 --seed=1 --tensorboard --epochs=200 --batch_size=4 --batches_train=125 --batches_test=15 --test_batch_size=20 --num_shots_test=15 --num_ways=5 --num_shots_train=1 --use_val_set --test_start=50 --val_start=50 --max_batches_process=2 --max_batches_process_test=5 --dataset=MiniImagenet --hidden_size=32 --param_init=kaiming --optimizer_slow=ADAM --lr_slow=0.001 --momentum_slow=0.9 --clamp_grads=0 --optimizer_fast=SGD --momentum_fast=0.9 --lr_fast=0.01 --wd_fast=0.5 --symmetric_ep --steps_first_phase=100 --polyak_avg --polyak_start=49 --steps_sec_phase=75 --steps_third_phase=75 --beta=0.01 --beta_annealing_rate=0 --beta_decay_rate=10 --outer_loop_decay=0.1 --outer_loop_step_scheduler=200 --test_after 1


# Omniglot calls

# 1-shot 5-ways
python3 train.py --batch_size=32 --steps_first_phase_test=200 --debug_iter=1 --seed=1 --tensorboard --epochs=30 --batches_train=125 --batches_test=15 --test_batch_size=20 --num_shots_test=15 --num_ways=5 --num_shots_train=1 --stride=1 --max_batches_process=8 --max_batches_process_test=10 --dataset=Omniglot --hidden_size=64 --param_init=kaiming --optimizer_slow=ADAM --lr_slow=0.01 --momentum_slow=0.9 --clamp_grads=0 --optimizer_fast=SGD --momentum_fast=0.9 --reset_optimiser --lr_fast=0.01 --wd_fast=0.5 --symmetric_ep --steps_first_phase=200 --steps_sec_phase=100 --steps_third_phase=100 --polyak_avg --polyak_start=4 --beta=0.1 --beta_annealing_rate=0 --beta_decay_rate=10 --outer_loop_decay=1. --outer_loop_step_scheduler=5


# 5-shot 5-ways
python3 train.py --batch_size=32 --steps_first_phase_test=200 --debug_iter=1 --seed=1 --tensorboard --epochs=30 --batches_train=125 --batches_test=15 --test_batch_size=20 --num_shots_test=15 --num_ways=5 --num_shots_train=5 --stride=1 --max_batches_process=8 --max_batches_process_test=10 --dataset=Omniglot --hidden_size=64 --param_init=kaiming  --optimizer_slow=ADAM --lr_slow=0.01 --momentum_slow=0.9 --clamp_grads=0 --optimizer_fast=SGD --momentum_fast=0.9 --reset_optimiser --lr_fast=0.01 --wd_fast=0.1 --symmetric_ep --steps_first_phase=200 --steps_sec_phase=100 --steps_third_phase=100 --polyak_avg --polyak_start=4 --beta=0.1 --beta_annealing_rate=0 --beta_decay_rate=10 --outer_loop_decay=1. --outer_loop_step_scheduler=5


# 1-shot 20-ways
python3 train.py --batch_size=16 --steps_third_phase=150 --steps_first_phase_test=200 --debug_iter=1 --seed=1 --tensorboard --epochs=30 --batches_train=125 --batches_test=15 --test_batch_size=20 --num_shots_test=15 --num_ways=20 --num_shots_train=1 --stride=1 --max_batches_process=8 --max_batches_process_test=10 --dataset=Omniglot --hidden_size=64 --param_init=kaiming --optimizer_slow=ADAM --lr_slow=0.01 --momentum_slow=0.9 --clamp_grads=0 --optimizer_fast=SGD --momentum_fast=0.9 --lr_fast=0.01 --wd_fast=0.5 --symmetric_ep --steps_first_phase=200 --steps_sec_phase=100 --steps_third_phase=100  --beta=0.03 --beta_annealing_rate=0 --beta_decay_rate=10 --outer_loop_decay=0.5 --outer_loop_step_scheduler=5 --polyak_avg --polyak_start=4 --reset_optimiser


# 5-shot 20-ways
python3 train.py --batch_size=16 --steps_first_phase_test=200 --debug_iter=1 --seed=1 --tensorboard --epochs=30 --batches_train=125 --batches_test=15 --test_batch_size=20 --num_shots_test=15 --num_ways=20 --num_shots_train=5 --stride=1 --max_batches_process=8 --max_batches_process_test=10 --dataset=Omniglot --hidden_size=64 --param_init=kaiming --reset_optimiser --lr_fast=0.01 --wd_fast=0.25 --symmetric_ep --steps_first_phase=200 --polyak_avg --optimizer_slow=ADAM --lr_slow=0.01 --momentum_slow=0.9 --clamp_grads=0 --optimizer_fast=SGD --momentum_fast=0.9 --steps_first_phase=200 --steps_sec_phase=200 --steps_third_phase=200 --polyak_avg --polyak_start=4 --beta=0.1 --beta_annealing_rate=0 --beta_decay_rate=10 --outer_loop_decay=1. --outer_loop_step_scheduler=5
