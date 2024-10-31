import os.path as osp

from visionsuite.engines.classification.src.datasets.default import load_data
from visionsuite.engines.classification.utils.registry import DATASETS


            

@DATASETS.register()
def directory_datasets(args):
    train_dir = osp.join(args.input_dir, "train")
    val_dir = osp.join(args.input_dir, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    
    return dataset, dataset_test, train_sampler, test_sampler