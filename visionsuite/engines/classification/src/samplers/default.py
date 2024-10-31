import torch 
from visionsuite.engines.classification.src.samplers.ra_sampler import RASampler
from visionsuite.engines.classification.utils.registry import SAMPLERS


@SAMPLERS.register()
def get_samplers(args, dataset, dataset_test):
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    return train_sampler, test_sampler