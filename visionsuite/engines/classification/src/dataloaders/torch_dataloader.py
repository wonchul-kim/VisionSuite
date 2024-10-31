import torch
from visionsuite.engines.classification.utils.registry import DATALOADERS


@DATALOADERS.register()
def torch_dataloader(train_dataset, train_sampler, batch_size, workers, collate_fn):
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return dataloader

