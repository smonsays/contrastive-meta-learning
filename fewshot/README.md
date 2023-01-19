# Code to reproduce our visual few-shot classification experiments on Omniglot and miniImageNet.

Hi! If you want to use our constrastive meta-learning aglorithm to learn vision models to do few-shot learning, you came to the right place. As a proof of concept, we experimented with standard benchmarks based on the Omniglot and miniImagenet dataset as well as small convolutional neural networks. 

Please refer to the [commands.sh](commands.sh)  file for commands to start the Omniglot and miniImageNet experiments that we reported in our paper. 

Please note: 

1. Unfortunately, when running the miniImageNet experiments, one has to download the data manually and put them in the folder "data_i". 
In the data_i folder you should have a miniimagenet folder that contains the following files: test_data.hdf5, test_labels.json, train_data.hdf5, train_labels.json, val_data.hdf5, val_labels.json
See issue with torchmeta https://github.com/tristandeleu/pytorch-meta/issues/134

2. We experimented with multiprocessing to parallelize computation over batches of data. Please adjust the arguments --max_batches_process --max_batches_process_test to adjust the usage of your GPU. The values used in commands.sh should enable running the experiments on 12GB GPUs. The very first time you execute the code, the multiprocessing might run into problems. Please just stop and reexecute. 

Please do not hesitate to contact us if you have questions, remarks or interesting things to discuss.


