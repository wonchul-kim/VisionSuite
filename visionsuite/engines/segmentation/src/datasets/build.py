

from visionsuite.engines.segmentation.utils.registry import DATASETS

def build_dataset(transform=None, **config):
    dataset = DATASETS.get(config['type'], case_sensitive=config['case_sensitive'])
    dataset(transform=transform)
    
    return dataset()    
    
    
# from visionsuite.engines.segmentation.src.datasets.coco_utils import get_coco
# from visionsuite.engines.segmentation.src.datasets.mask_dataset import get_mask
# import torchvision

# def get_dataset(args, is_train):
#     def sbd(*args, **kwargs):
#         kwargs.pop("use_v2")
#         return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

#     def voc(*args, **kwargs):
#         kwargs.pop("use_v2")
#         return torchvision.datasets.VOCSegmentation(*args, **kwargs)

#     paths = {
#         "voc": (args['input_dir'], voc, 21),
#         "voc_aug": (args['input_dir'], sbd, 21),
#         "coco": (args['input_dir'], get_coco, 21),
#         "mask": (args['input_dir'], get_mask, 4),
#     }
#     p, ds_fn, num_classes = paths[args['dataset']]

#     image_set = "train" if is_train else "val"
#     ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args['use_v2'])
#     return ds, num_classes
