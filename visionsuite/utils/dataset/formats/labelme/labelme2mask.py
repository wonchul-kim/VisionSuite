import os 
import os.path as osp 
import glob 
import json 
import cv2
from tqdm import tqdm

from visionsuite.utils.dataset.formats.labelme.utils import get_mask_from_labelme


def labelme2mask(input_dir, output_dir, modes, class2label, width=None, height=None, vis=False):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
    if vis:
        vis_dir = osp.join(output_dir, 'vis')
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)

    for mode in modes:
        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        for json_file in tqdm(json_files, desc=mode):
            filename = osp.split(osp.splitext(json_file)[0])[-1]
            with open(json_file, 'r') as jf:
                anns = json.load(jf)
                
            if width is None and height is None:
                width, height = anns['imageWidth'], anns['imageHeight']
            mask = get_mask_from_labelme(json_file, width, height, class2label, 
                                        format='opencv')
            
            cv2.imwrite(osp.join(_output_dir, filename + '.bmp'), mask)
            
            if vis:
                import numpy as np
                import imgviz 
                
                assert osp.exists(osp.join(input_dir, filename + '.bmp')), ValueError(f"Input Image at input_dir must exist")
                img = cv2.imread(osp.join(input_dir, filename + '.bmp'))
                
                vis_img = np.zeros((height, width*2, 3))
                vis_img[:, :width, :] = img
                color_map = imgviz.label_colormap(50)
                mask = color_map[mask.astype(np.uint8)].astype(np.uint8)
                vis_img[:, width:, :] = mask 
                
                cv2.imwrite(osp.join(vis_dir, filename + '.bmp'), vis_img)
        
            
if __name__ == '__main__':
    input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/split_datasets/mask/split_dataset/val/images'
    output_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/split_datasets/mask/split_dataset/val/masks'
    modes = ['./']
    # class2label = {'tear': 1}
    # class2label = {'scratch': 1}
    # class2label = {'scratch': 1, 'tear': 2}
    # class2label = {'scratch': 1, 'tear': 2, 'stabbed': 3}
    class2label = {'tear': 1, 'stabbed': 2}
    width, height = 512, 512
    vis = True

    labelme2mask(input_dir, output_dir, modes, class2label, width, height, vis)


