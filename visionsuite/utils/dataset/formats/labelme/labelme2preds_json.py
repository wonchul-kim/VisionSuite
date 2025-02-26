import os 
import os.path as osp 
import glob 
import json 
import cv2
from tqdm import tqdm
from shutil import copyfile
import numpy as np
from visionsuite.utils.helpers import get_filename

from visionsuite.utils.dataset.formats.labelme.utils import get_mask_from_labelme


def labelme2preds_json(input_dir, output_dir, idx2class):

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(osp.join(input_dir, '*.json'))
    
    results = {}
    for json_file in tqdm(json_files):
        filename = get_filename(json_file, False)
        img_file = osp.splitext(json_file)[0] + '.bmp'
        with open(json_file, 'r') as jf:
            anns = json.load(jf)['shapes']
            
        for ann in anns:
            idx2xyxys = {}
            for idx in idx2class.keys():
                if idx not in idx2xyxys:
                    idx2xyxys[idx] = {'polygon': []}
                
                idx2xyxys[idx]['polygon'].append(ann['points'])
            
            results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
                
    with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    input_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/m2f_100epochs/test/labels'
    output_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/m2f_100epochs/test/preds'
    classes = ['CHAMFER_MARK', 'LINE', 'MARK']
    idx2class = {idx: cls for idx, cls in enumerate(classes)}

    labelme2preds_json(input_dir, output_dir, idx2class)

