import torch
from .dist import is_main_process

def get_device(device):
    return torch.device(device)

def set_torch_deterministic(use_deterministic):
    if use_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    # if len(*batch) == 2:
    #     images, targets = list(zip(*batch))
    # elif len(*batch) == 3:
    #     images, targets, filenames = list(zip(*batch))
    # else:
    #     raise RuntimeError(f"You need to add output arguments at dataset. There are {len(*batch)} outputs")
    images, targets, filenames = list(zip(*batch))
    
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets, filenames


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

