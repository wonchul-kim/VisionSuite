import os 
import os.path as osp 
import glob 
import json 
import cv2

from visionsuite.utils.dataset.formats.labelme.utils import get_mask_from_labelme

input_dir = '/HDD/datasets/projects/22.03.22_flange_s/data'
output_dir = '/HDD/datasets/projects/22.03.22_flange_s/mask'
modes = ['./']
class2label = {'flange_s': 1}

if not osp.exists(output_dir):
    os.mkdir(output_dir)

for mode in modes:
    _output_dir = osp.join(output_dir, mode)
    if not osp.exists(_output_dir):
        os.mkdir(_output_dir)
    json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
    
    for json_file in json_files:
        filename = osp.split(osp.splitext(json_file)[0])[-1]
        with open(json_file, 'r') as jf:
            anns = json.load(jf)
            
        width, height = anns['imageWidth'], anns['imageHeight']
        channel = 3
        mask = get_mask_from_labelme(json_file, width, height, class2label, 
                                     format='opencv')
        
        cv2.imwrite(osp.join(_output_dir, filename + '.bmp'), mask)
        
        

                
            