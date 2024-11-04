from .cifar_dataset import cifar10_datasets
from .default import get_datasets
from .load_data import load_data, image_folder #TODO: to be deprecated

__all__ = ['get_datasets', 'cifar10_datasets', 'load_data', 'image_folder']