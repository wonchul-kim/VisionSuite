import sys

import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from visionsuite.engines.data.curate.src.utils import setup_logging

from visionsuite.engines.data.curate.src.dist_comm import (
    enable_distributed,
    get_global_rank,
    get_global_size,
    is_main_process,
    synchronize,
)
from visionsuite.engines.data.curate.src import distributed_kmeans_gpu as dkmg, kmeans_gpu as kmg


logger = logging.getLogger("hkmeans")


def split_clusters(
    data_path,
    subset_indices_path,
    clusters_path,
    n_splits,
    n_iters,
    dtype,
    high_precision,
    save_path,
    device="cuda",
    use_torchrun=False,
    checkpoint_period=10,
    verbose=False,
):
    enable_distributed(
        use_torchrun=use_torchrun,
        overwrite=True,
    )
    import os.path as osp 
    from glob import glob 
    from tqdm import tqdm

    folders = [folder.split("/")[-1] for folder in glob(osp.join(data_path, "**")) if not osp.isfile(folder)]

    embeddings = []
    for folder in tqdm(folders):
        embedding_files = glob(osp.join(data_path, folder, 'dinov2-large/attention_False', '*.npy'))
        for embedding_file in embedding_files:
            embeddings.append(np.load(embedding_file, mmap_mode='r'))

    try:
        X = np.concatenate(embeddings, axis=0)
        print("******** X.shape: ", X.shape) # 668985
    except:
        X = np.load(data_path, mmap_mode="r")
    
    if subset_indices_path is not None:
        logger.info(f"Using subset with indices in {subset_indices_path}")
        subset_indices = np.load(subset_indices_path)
        X = dkmg.ExtendedNumpyMemMap(X, subset_indices)
    clusters = np.load(clusters_path, allow_pickle=True)
    n_clusters = len(clusters)

    part_indices = dkmg.get_part_indices(n_clusters, get_global_size())
    rank = get_global_rank()

    # load checkpoints if exist
    if Path(save_path, f"split_checkpoint_{rank}.npy").exists():
        ckpt = np.load(
            Path(save_path, f"split_checkpoint_{rank}.npy"), allow_pickle=True
        ).item()
        small_centroids = list(ckpt["small_centroids"])
        small_clusters = list(ckpt["small_clusters"])
        last_index = ckpt["last_index"]
        assert last_index - part_indices[rank] + 1 == len(small_centroids)
    else:
        small_centroids = []
        small_clusters = []
        last_index = part_indices[rank] - 1

    # run kmeans++ on clusters
    for cluster_idx in tqdm(
        range(last_index + 1, part_indices[rank + 1]),
        desc="Splitting pre-clusters",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        if verbose:
            logger.info(f"Processing cluster {cluster_idx}")
        point_indices = np.sort(clusters[cluster_idx])
        if len(point_indices) > 0:
            point_feats = torch.tensor(X[point_indices], device=device, dtype=dtype)
            _small_centroids, _small_clusters, _, _ = kmg.kmeans(
                point_feats,
                min(n_splits, len(point_indices)),
                n_iters,
                chunk_size=-1,
                init_method="kmeans++",
                dist="l2",
                high_precision=high_precision,
            )

            _small_clusters = kmg.sort_cluster_by_distance(
                point_feats,
                _small_centroids,
                _small_clusters,
                device="cuda",
                dtype=dtype,
            )
            _small_clusters = [point_indices[el.astype(int)] for el in _small_clusters]

            non_empty_clusters = [len(el) > 0 for el in _small_clusters]
            _small_clusters = [el for el in _small_clusters if len(el) > 0]
            _small_centroids = _small_centroids[non_empty_clusters]

            small_centroids.append(_small_centroids.cpu().numpy())
            small_clusters += _small_clusters

            del point_feats
        if(
            cluster_idx % checkpoint_period == 0 or
            cluster_idx == part_indices[rank + 1] - 1
        ):
            np.save(
                Path(save_path, f"split_checkpoint_{rank}.npy"),
                {
                    "small_centroids": small_centroids,
                    "small_clusters": small_clusters,
                    "last_index": cluster_idx,
                },
            )
    synchronize()
    logger.info("Gathering clusters")
    if is_main_process():
        centroids = []
        clusters = []
        for i in tqdm(
            range(get_global_size()),
            desc="Gathering splitted clusters",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            split_data = np.load(
                Path(save_path, f"split_checkpoint_{i}.npy"),
                allow_pickle=True
            ).item()
            small_centroids = np.concatenate(split_data["small_centroids"])
            small_clusters = split_data["small_clusters"]
            assert(
                len(small_centroids) == len(small_clusters)
            ), f"Inconsistent shape in split_checkpoint_{i}.npy"
            assert split_data["last_index"] == part_indices[i + 1] - 1
            centroids.append(small_centroids)
            clusters += small_clusters
        centroids = np.concatenate(centroids)
        clusters = np.array(clusters, dtype=object)

        logger.info("Saving centroids and clusters")
        np.save(Path(save_path, "centroids.npy"), centroids)
        np.save(Path(save_path, "sorted_clusters.npy"), clusters)
        logger.info("Cleaning checkpoints")
        for i in range(get_global_size()):
            Path(save_path, f"split_checkpoint_{i}.npy").unlink(missing_ok=True)
    logger.info("Finished split_clusters!")

if __name__ == "__main__":
    import argparse
    import os.path as osp 
    import yaml 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='/HDD/_projects/github/VisionSuite/visionsuite/engines/data/curate/data/unit_test/split_clusters.yaml')
    parser.add_argument('--use_torchrun', action="store_true")
    args = parser.parse_args()

    assert osp.exists(args.config), ValueError(f'There is no such config file at {args.config}')
    with open(args.config, 'r') as yf:
        config = yaml.load(yf)
    
    config = argparse.Namespace(**config)
    config.use_torchrun = args.use_torchrun


    import os
    os.makedirs(config.save_path, exist_ok=True)
    setup_logging()

    def parse_dtype(dtype):
        if dtype == "float32":
            return torch.float32
        elif dtype == "float64":
            return torch.float64
        elif dtype == "float16":
            return torch.float16
        else:
            raise ValueError(f"Value of config.dtype ({config.dtype}) not regconised")

    config.dtype = parse_dtype(config.dtype)
    config.high_precision = parse_dtype(config.high_precision)

    split_clusters(
        config.data_path,
        config.subset_indices_path,
        config.clusters_path,
        config.n_splits,
        config.n_iters,
        config.dtype,
        config.high_precision,
        config.save_path,
        "cuda",
        config.use_torchrun,
    )
