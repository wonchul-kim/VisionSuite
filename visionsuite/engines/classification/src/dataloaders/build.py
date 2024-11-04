from visionsuite.engines.classification.utils.registry import DATALOADERS


def build_dataloader(args, dataset, collate_fn):
    dataloader = DATALOADERS.get('torch_dataloader')(dataset.train_dataset, dataset.train_sampler, args['batch_size'], args['workers'], collate_fn)

    return dataloader