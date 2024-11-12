from visionsuite.engines.utils.torch_utils.utils import collate_fn
from visionsuite.engines.segmentation.utils.registry import DATALOADERS

def build_dataloader(dataset, mode, **config):
    try:
        dataloader = DATALOADERS.get(config['type'], case_sensitive=config['case_sensitive'])(dataset=getattr(dataset, f'{mode}_dataset'), 
                                                                                                sampler=getattr(dataset, f'{mode}_sampler'), 
                                                                                                batch_size=config[mode]['batch_size'], 
                                                                                                workers=config[mode]['workers'])
    except Exception as error:
        raise Exception(f"{error} at build_dataloader")
    
    return dataloader


# def get_dataloader(args, dataset, dataset_test):
#     if args['distributed'] and args['distributed']['use']:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
#     else:
#         train_sampler = torch.utils.data.RandomSampler(dataset)
#         test_sampler = torch.utils.data.SequentialSampler(dataset_test)

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args['batch_size'],
#         sampler=train_sampler,
#         num_workers=args['workers'],
#         collate_fn=collate_fn,
#         drop_last=True,
#     )

#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1, sampler=test_sampler, num_workers=args['workers'], collate_fn=collate_fn
#     )
    
#     return data_loader, data_loader_test, train_sampler, test_sampler