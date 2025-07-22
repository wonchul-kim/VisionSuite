import os.path as osp 
import glob
from PIL import Image 
from visionsuite.engines.utils.torch_utils.torch_dist_env import get_global_rank, get_global_size, synchronize



def get_part_indices(num_points, world_size):
    """
    Get indices of data points managed by each worker
    """
    return [round(num_points / world_size * i) for i in range(world_size + 1)]

class LabelmeDataset:
    def __init__(self, root, transform, roi=[], image_format='bmp', search_all=False):
        self.transform = transform
        self.roi = roi
        
        
        
        if search_all:
            self.img_files = glob.glob(osp.join(root, f'*/*.{image_format}'))
        else:
            self.img_files = glob.glob(osp.join(root, f'*.{image_format}'))
        # self.img_files = self.img_files[:10]
        print(f"There are {len(self.img_files)} image files")
        rank = get_global_rank()
        part_indices = get_part_indices(len(self.img_files), get_global_size())
        print(f"Rank {rank}: Loading data")
        print(f"part_indices: {part_indices}")
        self.img_files = self.img_files[part_indices[rank] : part_indices[rank + 1]]
        print(f"After slicing, there are {len(self.img_files)} image files")
        synchronize()
        
        
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file)
        if self.roi != [] and len(self.roi) == 4:
            img = img.crop((self.roi[0], self.roi[1], self.roi[2], self.roi[3]))

        if self.transform:
            img = self.transform(img)
            
        return img, osp.split(osp.splitext(img_file)[0])[-1]
        
        
    def __len__(self):
        return len(self.img_files)