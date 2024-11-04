from visionsuite.engines.classification.utils.registry import DATALOADERS


def build_dataloader(args, dataset, mode='train'):
    try:
        dataloader = DATALOADERS.get('torch_dataloader')(dataset=getattr(dataset, f'{mode}_dataset'), 
                                                     sampler=getattr(dataset, f'{mode}_sampler'), 
                                                     batch_size=args['batch_size'], 
                                                     workers=args['workers'],
                                                     augment=args['augment'])
    except Exception as error:
        raise Exception(f"{error} at build_dataloader")
    
    return dataloader