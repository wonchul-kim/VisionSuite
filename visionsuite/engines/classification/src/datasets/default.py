from visionsuite.engines.classification.utils.registry import DATASETS


@DATASETS.register()
def get_datasets(args):   
    if args.dataset == 'directory':
        return DATASETS.get("directory_datasets")(args)
    
    elif args.dataset == 'cifar10':
        return DATASETS.get("cifar10_datasets")(args)
    
    else:
        raise NotImplementedError(f"There is no such dataset module for {args.dataset}")


