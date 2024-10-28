import os.path as osp

from visionsuite.engines.classification.datasets.default import load_data
from visionsuite.engines.classification.samplers.default import get_samplers

def get_datasets(args):
    
    
    if args.dataset == 'directory':
        return get_directory_datasets(args)
    
    elif args.dataset == 'cifar10':
        return get_cifar10_datasets(args)
    
    else:
        raise NotImplementedError(f"There is no such dataset module for {args.dataset}")
    
    
def get_cifar10_datasets(args):
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./tmp/data', 
                                                 train=True, download=True, transform=transform)
    
    val_dataset = torchvision.datasets.CIFAR10(root='./tmp/data', 
                                                 train=False, download=True, transform=transform)
    
    train_sampler, val_sampler = get_samplers(args, train_dataset, val_dataset)
    
    
    return train_dataset, val_dataset, train_sampler, val_sampler
        



def get_directory_datasets(args):
    train_dir = osp.join(args.input_dir, "train")
    val_dir = osp.join(args.input_dir, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    
    return dataset, dataset_test, train_sampler, test_sampler