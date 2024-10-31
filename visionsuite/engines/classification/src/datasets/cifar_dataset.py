import torchvision
import torchvision.transforms as transforms

from visionsuite.engines.classification.src.samplers.default import get_samplers
from visionsuite.engines.classification.utils.registry import DATASETS


@DATASETS.register()
def cifar10_datasets(args):

    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./tmp/data', 
                                                 train=True, download=True, transform=transform)
    
    val_dataset = torchvision.datasets.CIFAR10(root='./tmp/data', 
                                                 train=False, download=True, transform=transform)
    
    train_sampler, val_sampler = get_samplers(args, train_dataset, val_dataset)
    
    
    return train_dataset, val_dataset, train_sampler, val_sampler
        
