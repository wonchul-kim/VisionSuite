
# [Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach](https://arxiv.org/abs/2405.15613)

-----------------------------------------------------------------------------
## How to run 

1. `cluster.py`
```shell
torchrun --nnodes=1 --nproc_per_node=<num_of_gpus> visionsuite/engines/data/curate/cluster.py --config <path_to_config_file> --use_torchrun
```

* For the `config` file, refer to `visionsuite/engines/data/curate/data/unit_test/cluster.yaml`

2. `split_cluters.py`
```shell
torchrun --nnodes=1 --nproc_per_node=<num_of_gpus> visionsuite/engines/data/curate/split_clusters.py --config <path_to_config_file> --use_torchrun
```

* For the `config` file, refer to `visionsuite/engines/data/curate/data/unit_test/split_clusters.yaml`

-----------------------------------------------------------------------------
## References

- [Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach](https://arxiv.org/abs/2405.15613) [code](https://github.com/facebookresearch/ssl-data-curation)