import os.path as osp 
import glob
from PIL import Image 

class LabelmeDataset:
    def __init__(self, root, transform, roi=[], image_format='bmp'):
        self.transform = transform
        self.roi = roi
        self.img_files = glob.glob(osp.join(root, f'*.{image_format}'))
        # self.img_files = self.img_files[:10]
        print(f"There are {len(self.img_files)} image files")
        
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