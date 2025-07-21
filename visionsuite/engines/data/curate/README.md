



1. `cluster.py`
```shell
torchrun --nnodes=1 --nproc_per_node=<num_of_gpus> visionsuite/engines/data/curate/cluster.py --use_torchrun
```

2. `split_cluters.py`