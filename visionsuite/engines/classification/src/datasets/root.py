from .cifar_dataset import cifar10_datasets
from .directory_dataset import torchivision_image_folder_dataset 
from .default import get_datasets
from .load_data import load_data #TODO: to be deprecated

__all__ = ['get_datasets', 'cifar10_datasets', 'torchivision_image_folder_dataset', 'load_data']