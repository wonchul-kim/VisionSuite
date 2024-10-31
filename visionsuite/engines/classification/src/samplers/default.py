import torch 
from visionsuite.engines.classification.src.samplers.ra_sampler import RASampler
from visionsuite.engines.classification.utils.registry import SAMPLERS


@SAMPLERS.register()
def get_samplers(args, train_dataset, val_dataset):
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)
        
    return train_sampler, test_sampler