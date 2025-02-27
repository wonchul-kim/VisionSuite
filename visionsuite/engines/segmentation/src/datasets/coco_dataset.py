import copy
import os

import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from visionsuite.engines.utils.torch_utils.transforms import Compose


from visionsuite.engines.segmentation.utils.registry import DATASETS
from visionsuite.engines.segmentation.src.datasets.base_dataset import BaseDataset

import torchvision
from torchvision.transforms import functional as F, InterpolationMode
import visionsuite.engines.utils.torch_utils.presets as presets

def get_transform(is_train, args):
    if is_train:
        return presets.SegmentationPresetTrain(base_size=512, crop_size=480, backend=args['backend'], use_v2=args['use_v2'])
    elif args['weights'] and args['test_only']:
        weights = torchvision.models.get_weight(args['weights'])
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=512, backend=args['backend'], use_v2=args['use_v2'])

import copy
import os

import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from visionsuite.engines.utils.torch_utils.transforms import Compose


class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset



def get_dataset(root, image_set, transforms, use_v2=False):
    
    PATHS = {
                "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
                "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
                # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
            }
    # CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    CAT_LIST = [0, 5, 2]

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    # The 2 "Compose" below achieve the same thing: converting coco detection
    # samples into segmentation-compatible samples. They just do it with
    # slightly different implementations. We could refactor and unify, but
    # keeping them separate helps keeping the v2 version clean
    if use_v2:
        import visionsuite.engines.utils.torch_utils.v2_extras as v2_extras
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        transforms = Compose([v2_extras.CocoDetectionToVOCSegmentation(), transforms])
        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys={"masks", "labels"})
    else:
        transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), transforms])
        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset, CAT_LIST

@DATASETS.register()
class COCODataset(BaseDataset):
    def __init__(self, transform=None):
        super().__init__(transform=transform)
        
    def load_dataset(self):
        super().load_dataset()
        self.train_dataset, self.classes = get_dataset(self.args['input_dir'], 'train', 
                                         get_transform(True, {'weights': None, 'test_only': False, 'backend': 'PIL', 'use_v2': False}))
        self.val_dataset, _ = get_dataset(self.args['input_dir'], 'val', 
                                         get_transform(True, {'weights': None, 'test_only': False, 'backend': 'PIL', 'use_v2': False}))
        
        
        self.label2index = {index: label for index, label in enumerate(self.classes)}
        self.index2label = {label: index for index, label in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        print(f"label2index: {self.label2index}")
