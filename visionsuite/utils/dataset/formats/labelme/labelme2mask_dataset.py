import os 
import os.path as osp 
import glob 
import json 
import cv2
from tqdm import tqdm
from shutil import copyfile
from visionsuite.utils.helpers import get_filename

from visionsuite.utils.dataset.formats.labelme.utils import get_mask_from_labelme


def labelme2mask(input_dir, output_dir, class2label, width=None, height=None, vis=False):

    modes = [folder.split("/")[-1] for folder in glob.glob(osp.join(input_dir, "**")) if not osp.isfile(folder)]

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for mode in modes:
        _output_dir = osp.join(output_dir, mode)
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
            
            
        if vis:
            vis_dir = osp.join(_output_dir, 'vis')
            if not osp.exists(vis_dir):
                os.mkdir(vis_dir)
            
        json_files = glob.glob(osp.join(input_dir, mode, '*.json'))
        
        for json_file in tqdm(json_files, desc=mode):
            img_file = osp.splitext(json_file)[0] + '.bmp'
            filename = get_filename(json_file, False)
            with open(json_file, 'r') as jf:
                anns = json.load(jf)
                
            if width is None and height is None:
                width, height = anns['imageWidth'], anns['imageHeight']
            mask = get_mask_from_labelme(json_file, width, height, class2label, 
                                        format='opencv')
            
            mask_output_dir = osp.join(_output_dir, 'masks')
            if not osp.exists(mask_output_dir):
                os.mkdir(mask_output_dir)
                        
            image_output_dir = osp.join(_output_dir, 'images')
            if not osp.exists(image_output_dir):
                os.mkdir(image_output_dir)
                
            cv2.imwrite(osp.join(mask_output_dir, filename + '.bmp'), mask)
            copyfile(img_file, osp.join(image_output_dir, filename + '.bmp'))
            
            if vis:
                import numpy as np
                import imgviz 
                
                img = cv2.imread(img_file)
                
                vis_img = np.zeros((height, width*2, 3))
                vis_img[:, :width, :] = img
                color_map = imgviz.label_colormap(50)
                mask = color_map[mask.astype(np.uint8)].astype(np.uint8)
                vis_img[:, width:, :] = mask 
                
                cv2.imwrite(osp.join(vis_dir, filename + '.bmp'), vis_img)
        
            
if __name__ == '__main__':
    input_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/tenneco/split_patch_labelme_dataset'
    output_dir = '/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/tenneco/split_patch_mask_dataset'
    # class2label = {'tear': 1}
    # class2label = {'scratch': 1}
    # class2label = {'scratch': 1, 'tear': 2}
    # class2label = {'scratch': 1, 'tear': 2, 'stabbed': 3}
    class2label = {'stabbed': 1, 'stabbed_c': 2, 'stabbed_p': 3, 'scratch': 4}
    width, height = 512, 512
    vis = True

    labelme2mask(input_dir, output_dir, class2label, width, height, vis)


