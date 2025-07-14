import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from .labelme_dataset import LabelmeDataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")

def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)


def get_default_transforms():
    return transforms.Compose([
        # transforms.Resize((8652, 1022), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.Resize((768, 1120), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.Resize((350, 560), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        _convert_image_to_rgb,
        _safe_to_tensor,
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])


def get_dataloaders(dataset, transform, batch_size, root_dir, roi=[]):
    if transform is None:
        # just dummy resize -> both CLIP and DINO support 224 size of the image
        transform = get_default_transforms()
    dataset = get_datasets(dataset, transform, root_dir, roi=roi)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    return dataloader


def get_datasets(dataset, transform, root_dir, roi=[]):
    data_path = os.path.join(root_dir)#, 'datasets')

    # if dataset == 'cifar10':
    #     train_dataset = dsets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    #     val_dataset = dsets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    # elif dataset == 'cifar100':
    #     train_dataset = dsets.CIFAR100(root=data_path, train=True, transform=transform, download=True)
    #     val_dataset = dsets.CIFAR100(root=data_path, train=False, transform=transform, download=True)

    # elif dataset == 'folder':
    #     train_dataset = dsets.ImageFolder(root=os.path.join(data_path, "train"), transform=transform)
    #     val_dataset = dsets.ImageFolder(root=os.path.join(data_path, "val"), transform=transform)
        
    # elif dataset == "imagenet":
    #     '''
    #         Manually download from https://www.image-net.org/ and put to ./data/datasets/imagenet
    #     '''
    #     train_dataset = dsets.ImageFolder(root=os.path.join(data_path, "imagenet/train"), transform=transform)
    #     val_dataset = dsets.ImageFolder(root=os.path.join(data_path, "imagenet/val"), transform=transform)
        
    # elif dataset == 'labelme':
    dataset = LabelmeDataset(root=data_path, transform=transform, roi=roi)

    return dataset

