import glob 
import os
import os.path as osp

import imgviz
import json
import cv2
import numpy as np
import pandas as pd
from visionsuite.utils.visualizers.vis_seg import vis_seg
from athena.src.tasks.segmentation.frameworks.tensorflow.models.tf_model_v2 import TFModelv2


def get_mask_from_pred(pred, conf=0.5, contour_thres=50):
    mask = (pred > conf).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        if len(contour) < contour_thres:
            pass
        else:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].tolist()
            points.append(approx)

    return mask, points

patch_overlap_ratio = 0.2
patch_width = 512
patch_height = 512
dx = int((1. - patch_overlap_ratio) * patch_width)
dy = int((1. - patch_overlap_ratio) * patch_height)

compare_mask = True
weights = '/HDD/_projects/benchmark/semantic_segmentation/sungwoo_bottom/outputs/outputs/SEGMENTATION/w_patch_ratio_0.4_1_1_300/train/weights/last_weights.h5'
model = TFModelv2(model_name='deeplabv3plus', backbone='efficientnetb0', backbone_weights='imagenet', 
                  batch_size=1, width=patch_width, height=patch_height, channel=3, num_classes=4,
                  weights=weights)
classes = ['SCRATCH', 'TEAR', 'STABBED']

idx2class = {idx: cls for idx, cls in enumerate(classes)}
_classes = ['SCRATCH', 'TEAR', 'STABBED']
_idx2class = {idx: cls for idx, cls in enumerate(_classes)}
input_dir = '/HDD/_projects/benchmark/semantic_segmentation/sungwoo_bottom/datasets/split_dataset/val'
json_dir = '/HDD/_projects/benchmark/semantic_segmentation/sungwoo_bottom/datasets/split_dataset/val'
output_dir = '/HDD/_projects/benchmark/semantic_segmentation/sungwoo_bottom/tests/w_patch_ratio_0.4_1_1_300'
font_scale = 0

if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_files = glob.glob(osp.join(input_dir, '*.bmp'))

results = {}
compare = {}
for img_file in img_files:
    filename = osp.split(osp.splitext(img_file)[0])[-1]
    img = cv2.imread(img_file)
    img_h, img_w, img_c = img.shape

    num_patches = 0
    idx2xyxys = {}
    for y0 in range(0, img_h, dy):
        for x0 in range(0, img_w, dx):
            num_patches += 1
            if y0 + patch_height > img_h:
                y = img_h - patch_height
            else:
                y = y0

            if x0 + patch_width > img_w:
                x = img_w - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height
            preds = model(img[np.newaxis, ymin:ymax, xmin:xmax, :])[0].numpy()

            for idx in idx2class.keys():
                mask, points = get_mask_from_pred(preds[:, :, idx + 1], conf=0.6)
                
                
                for _points in points:
                    for _point in _points:
                        _point[0] += xmin
                        _point[1] += ymin
                
                    if idx not in idx2xyxys:
                        idx2xyxys[idx] = {'polygon': []}
                    
                    idx2xyxys[idx]['polygon'].append(_points)
                
            # if _classes is not None:
            #     new_idx2xyxys = {}
            #     for idx, _cls in enumerate(_classes):
            #         for jdx, cls in enumerate(classes):
            #             if cls == _cls:
            #                 new_idx2xyxys[idx] = idx2xyxys[jdx]
                
            #     idx2xyxys = new_idx2xyxys
            #     idx2class = _idx2class    
    
    color_map = imgviz.label_colormap()[1:len(idx2class) + 1]
    results.update({filename: {'idx2xyxys': idx2xyxys, 'idx2class': idx2class, 'img_file': img_file}})
    
    if compare_mask:
        _compare = vis_seg(img_file, idx2xyxys, idx2class, output_dir, color_map=color_map, json_dir=json_dir, 
                            seg_type='semantic', font_scale=font_scale,
                            compare_mask=compare_mask)
        _compare.update({"img_file": img_file})
        compare.update({filename: _compare})
    else:
        vis_seg(img_file, idx2xyxys, idx2class, output_dir, color_map, json_dir, compare_mask=compare_mask)
    
with open(osp.join(output_dir, 'preds.json'), 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

if compare_mask:
    with open(osp.join(output_dir, 'diff.json'), 'w', encoding='utf-8') as json_file:
        json.dump(compare, json_file, ensure_ascii=False, indent=4)
    
    df_compare = pd.DataFrame(compare)
        
    df_compare_pixel = df_compare.loc['diff_pixel'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_pixel.csv'))
    
    df_compare_pixel = df_compare.loc['diff_iou'].T
    df_compare_pixel.fillna(0, inplace=True)
    df_compare_pixel.to_csv(osp.join(output_dir, 'diff_iou.csv'))