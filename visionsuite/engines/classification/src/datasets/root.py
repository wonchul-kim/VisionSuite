from .cifar_dataset import cifar10_datasets
from .directory_dataset import directory_datasets 
from .default import get_datasets

__all__ = ['get_datasets', 'cifar10_datasets', 'directory_datasets']