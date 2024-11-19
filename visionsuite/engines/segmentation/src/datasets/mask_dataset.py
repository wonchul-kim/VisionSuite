import os
import os.path as osp
import torch 
import glob 
from PIL import Image 

from visionsuite.engines.segmentation.utils.registry import DATASETS
from visionsuite.engines.classification.src.datasets.base_dataset import BaseDataset

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
    


@DATASETS.register()
class MaskDatasetWrapper(BaseDataset):
    def __init__(self, name="MaskDatasetWrapper", transform=None):
        if transform is None:
            mean=(0.485, 0.456, 0.406)
            std=(0.229, 0.224, 0.225)
                
            import torchvision.transforms as transforms
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
            
        super().__init__(name=name, transform=transform)

    def load_dataset(self, train_folder_name='train', val_folder_name='val'):
        super().load_dataset()
        
        self.train_dataset =  DATASETS.get(self.args['load_dataset']['type'], 
                                           case_sensitive=self.args['load_dataset']['case_sensitive']
                                )(osp.join(self.args['input_dir'], train_folder_name), 
                                    get_transform(True, {"weights": None, "test_only": False, "backend": 'PIL', "use_v2": False}))
                                    # self._transform)
        self.log_info(f"LOADED train_dataset: {self.args['load_dataset']['type']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- input_dir: {self.args['input_dir']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- number of images: {len(self.train_dataset)}", self.build.__name__, __class__.__name__)
        self.log_info(f"- transforms: TODO", self.build.__name__, __class__.__name__)
        
        self.val_dataset =  DATASETS.get(self.args['load_dataset']['type'], 
                                         case_sensitive=self.args['load_dataset']['case_sensitive']
                                )(osp.join(self.args['input_dir'], val_folder_name), 
                                  get_transform(False, {"weights": None, "test_only": False, "backend": 'PIL', "use_v2": False}))
                                    # self._transform)
        self.log_info(f"LOADED val_dataset: {self.args['load_dataset']['type']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- input_dir: {self.args['input_dir']}", self.build.__name__, __class__.__name__)
        self.log_info(f"- number of images:: {len(self.val_dataset)}", self.build.__name__, __class__.__name__)
        self.log_info(f"- transforms: TODO", self.build.__name__, __class__.__name__)
        
        self.label2index = {index: label for index, label in enumerate(self.classes)}
        self.index2label = {label: index for index, label in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.log_info(f"- label2index: {self.label2index}", self.load_dataset.__name__, __class__.__name__)
     
        
@DATASETS.register()
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, transform=None, img_exts=['png', 'bmp']):
        self.input_dir = input_dir
        self.transform = transform

        self.img_files = []
        for img_ext in img_exts:
            self.img_files += glob.glob(os.path.join(self.input_dir, "images/*.{}".format(img_ext)))
        print(f"  - There are {len(self.img_files)} image files")
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): 
        print(idx)
        img_file = self.img_files[idx]
        fname = osp.split(osp.splitext(img_file)[0])[-1]

        mask_file = osp.join(self.input_dir, 'masks/{}.bmp'.format(fname))
        assert osp.exists(mask_file), RuntimeError(f"There is no such mask image: {mask_file}")

        image = Image.open(img_file)
        mask = Image.open(mask_file)        
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, fname
    
    


# import os.path as osp
# import glob 
# import os
# from PIL import Image
# import torch
# from visionsuite.engines.utils.torch_utils.transforms import Compose


# def get_mask(root, image_set, transforms, use_v2=False):

#     PATHS = {
#         "train": ("train/images"),
#         "val": ("val/images"),
#     }

#     transforms = Compose([transforms])

#     img_folder = PATHS[image_set]
#     img_folder = osp.join(root, img_folder)

#     dataset = MaskDataset(img_folder, transforms=transforms)

#     # if mode == "train": #FIXME: Need to make this option 
#     #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

#     return dataset


# ##FIXME: This maskdataset is too dependent to camvid dataset..., especially total_classes w/o background label 
# class MaskDataset(torch.utils.data.Dataset):
#     def __init__(self, img_folder, transforms=None, img_exts=['png', 'bmp']):
#         self.img_folder = img_folder
#         self.transforms = transforms

#         self.img_files = []
#         for img_ext in img_exts:
#             self.img_files += glob.glob(os.path.join(self.img_folder, "*.{}".format(img_ext)))
#         print(f"  - There are {len(self.img_files)} image files")
    
#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx): 
#         img_file = self.img_files[idx]
#         fname = osp.split(osp.splitext(img_file)[0])[-1]

#         mask_file = osp.join(self.img_folder, '../masks/{}.bmp'.format(fname))
#         assert osp.exists(mask_file), RuntimeError(f"There is no such mask image: {mask_file}")

#         image = Image.open(img_file)
#         mask = Image.open(mask_file)        
        
#         if self.transforms is not None:
#             image, mask = self.transforms(image, mask)

#         return image, mask, fname