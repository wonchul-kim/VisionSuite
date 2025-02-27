from glob import glob 
import os.path as osp 
import os 
import json
import numpy as np
import cv2
from tqdm import tqdm

def create_crops(input_dir, output_dir, offset):

    output_dir = osp.join(output_dir, f'offset_{offset}')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    img_files = glob(osp.join(input_dir, '*.bmp'))
    print(f"There are {len(img_files)} images at {input_dir}")
    for img_file in tqdm(img_files):
        filename = osp.split(osp.splitext(img_file)[0])[-1]
        json_file = osp.splitext(img_file)[0] + '.json'
        
        assert osp.exists(json_file), ValueError(F"There is no such json file: {json_file}")
        
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
        
        img = cv2.imread(img_file)    
        img_h, img_w, img_c = img.shape
        
        for idx, ann in enumerate(anns):
            label = ann['label']
            
            _output_dir = osp.join(output_dir, label)
            if not osp.exists(_output_dir):
                os.mkdir(_output_dir)
            
            shape_type = ann['shape_type']
            points = np.array(ann['points'])
            
            if shape_type == 'point' or len(points) == 1:
                continue
            
            xs, ys = points[:, 0], points[:, 1]
            
            if offset == 'auto':
                _offset = max((max(xs) - min(xs))/2, (max(ys) - min(ys))/2)
            else:
                _offset = offset
            
            roi = [min(xs) - _offset, min(ys) - _offset, max(xs) + _offset, max(ys) + _offset] # x1, y1, x2, y2
            
            if roi[0] <= 0:
                roi[0] = 0
                
            if roi[1] <= 0:
                roi[1] = 0
            
            if roi[2] >= img_w:
                roi[2] = img_w
                
            if roi[3] >= img_h:
                roi[3] = img_h
                
            roi = list(map(int, roi))
            crop = img[roi[1]:roi[3], roi[0]:roi[2]]
            
            cv2.imwrite(osp.join(_output_dir, filename + f'_{idx}.png'), crop)
            
    
if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_dataset/train'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/crop/defects'
    offset = 'auto'
    # offset = 100

    create_crops(input_dir, output_dir, offset)