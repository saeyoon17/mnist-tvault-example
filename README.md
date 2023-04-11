# mnist-tvault-example
Sample repository for classifying MNIST dataset with tvault model registry. 

## Description
This repository uses ResNet-18 model to classify MNIST dataset.
Since the purpose of the repository is to let you experience model registry, we set train epoch to 5.
In order to run experiment, run 

`python -u -m torch.distributed.launch --nproc_per_node {gpu_num} --use_env train.py`.

After experiment end, model registry will be created using statement in `train.py`

`tvault.log_all(model, tag=f"{tag}", result=f'{result}, optimizer=optimizer)`.



