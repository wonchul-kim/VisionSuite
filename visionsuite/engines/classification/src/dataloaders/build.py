from visionsuite.engines.classification.utils.registry import DATALOADERS


def build_dataloader(args, dataset, mode='train'):
    try:
        dataloader = DATALOADERS.get('torch_dataloader')(dataset=getattr(dataset, f'{mode}_dataset'), 
                                                     sampler=getattr(dataset, f'{mode}_sampler'), 
                                                     batch_size=args[mode]['batch_size'], 
                                                     workers=args[mode]['workers'],
                                                     mixup_cutmix=args['augment']['mixup_cutmix'])
    except Exception as error:
        raise Exception(f"{error} at build_dataloader")
    
    return dataloader