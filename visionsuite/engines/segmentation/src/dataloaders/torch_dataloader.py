import torch
from visionsuite.engines.segmentation.utils.registry import DATALOADERS
from torch.utils.data.dataloader import default_collate


@DATALOADERS.register()
def torch_dataloader(dataset, sampler, batch_size, workers, mixup_cutmix=None):
    
    collate_fn = default_collate
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return dataloader

