import torch

def get_dataloader(dataset, dataset_test, train_sampler, test_sampler, batch_size, workers, collate_fn):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=workers, pin_memory=True
    )
    
    return data_loader, data_loader_test