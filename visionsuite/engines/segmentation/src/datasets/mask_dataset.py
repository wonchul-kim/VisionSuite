import os.path as osp
import glob 
import os
from PIL import Image
import torch
from visionsuite.engines.utils.torch_utils.transforms import Compose


def get_mask(root, image_set, transforms, use_v2=False):

    PATHS = {
        "train": ("train/images"),
        "val": ("val/images"),
    }

    transforms = Compose([transforms])

    img_folder = PATHS[image_set]
    img_folder = osp.join(root, img_folder)

    dataset = MaskDataset(img_folder, transforms=transforms)

    # if mode == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


##FIXME: This maskdataset is too dependent to camvid dataset..., especially total_classes w/o background label 
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, img_exts=['png', 'bmp']):
        self.img_folder = img_folder
        self.transforms = transforms

        self.img_files = []
        for img_ext in img_exts:
            self.img_files += glob.glob(os.path.join(self.img_folder, "*.{}".format(img_ext)))
        print(f"  - There are {len(self.img_files)} image files")
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): 
        img_file = self.img_files[idx]
        fname = osp.split(osp.splitext(img_file)[0])[-1]

        mask_file = osp.join(self.img_folder, '../masks/{}.bmp'.format(fname))
        assert osp.exists(mask_file), RuntimeError(f"There is no such mask image: {mask_file}")

        image = Image.open(img_file)
        mask = Image.open(mask_file)        
        
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, fname