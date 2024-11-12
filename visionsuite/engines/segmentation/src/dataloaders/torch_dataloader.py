import torch
from visionsuite.engines.segmentation.utils.registry import DATALOADERS
from torch.utils.data.dataloader import default_collate


@DATALOADERS.register()
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, 
                 num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, 
                 timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None,  
                 prefetch_factor=None, persistent_workers=False, pin_memory_device=''):
        
        if collate_fn is None:
            collate_fn = default_collate
            
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, 
                 num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, 
                 timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, generator=generator,  
                 prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
            