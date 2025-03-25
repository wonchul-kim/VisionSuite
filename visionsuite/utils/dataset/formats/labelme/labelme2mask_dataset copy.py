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


def labelme2mask(input_dir, output_dir, class2label, input_format, modes, 
                 width=None, height=None, vis=False, output_format='bmp', 
                 rois=[[]], one_class=False, crop={'use': False}):

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
            img_file = osp.splitext(json_file)[0] + f'.{input_format}'
            img = cv2.imread(img_file)
            filename = get_filename(json_file, False)
            with open(json_file, 'r') as jf:
                anns = json.load(jf)
                
            if width is None and height is None:
                width, height = anns['imageWidth'], anns['imageHeight']
                
            if rois == [[]]:
                rois = [[0, 0, width, height]]
            mask = get_mask_from_labelme(json_file, width, height, class2label, 
                                        format='opencv', one_class=one_class)
            import numpy as np
            mask_output_dir = osp.join(_output_dir, 'masks')
            if not osp.exists(mask_output_dir):
                os.mkdir(mask_output_dir)
                        
            image_output_dir = osp.join(_output_dir, 'images')
            if not osp.exists(image_output_dir):
                os.mkdir(image_output_dir)
                
            for roi in rois:
                roi_img = img[roi[1]:roi[3], roi[0]:roi[2]]
                roi_mask = mask[roi[1]:roi[3], roi[0]:roi[2]]
                
                if not crop['use']:
                    cv2.imwrite(osp.join(mask_output_dir, filename + f'.{output_format}'), roi_mask)
                    cv2.imwrite(osp.join(image_output_dir, filename + f'.{output_format}'), roi_img)
                    
                    if vis:
                        import numpy as np
                        import imgviz 
                        
                        vis_img = np.zeros((roi[3] - roi[1], (roi[2] - roi[0])*2, 3))
                        vis_img[:, :roi[2] - roi[0], :] = roi_img
                        if one_class:
                            vis_img[:, roi_img.shape[1]:, :] = cv2.cvtColor(roi_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
                        else:
                            color_map = imgviz.label_colormap(50)
                            vis_img[:, roi[2] - roi[0]:, :] = color_map[roi_mask.astype(np.uint8)].astype(np.uint8) 
                        
                        cv2.imwrite(osp.join(vis_dir, filename + '.png'), vis_img)
                else:
                    
                    for ann_id, ann in enumerate(anns['shapes']):

                        if ann['shape_type'] in ['point', 'polygon', 'watershed'] and len(ann['points']) <=2 :
                            continue
                        xs, ys = [], []
                        for point in ann['points']:
                            if point[0] >= roi[0] and point[0] <= roi[2]:
                                xs.append(point[0] - roi[0])
                            elif point[0] < roi[0]:
                                xs.append(0)
                            elif point[0] > roi[2]:
                                xs.append(roi[2] - roi[0])
                            else:
                                raise RuntimeError(
                                    f"Not considered x: point is {point} and roi is {roi}"
                                )

                            if point[1] >= roi[1] and point[1] <= roi[3]:
                                ys.append(point[1] - roi[1])
                            elif point[1] < roi[1]:
                                ys.append(0)
                            elif point[1] > roi[3]:
                                ys.append(roi[3] - roi[1])
                            else:
                                raise RuntimeError(
                                    f"Not considered y: point is {point} and roi is {roi}"
                                )
                                
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        obj_width = max(xs) - min(xs)
                        obj_height = max(ys) - min(ys)
                        case = None
                        if isinstance(crop['offset'], float) and crop['offset'] <= 1 and crop['offset'] >= 0:
                            offset_x, offset_y = (x2 - x1) * crop['offset'], (y2 - y1) * crop['offset']
                            case = 1
                        elif isinstance(crop['offset'], (tuple, list)):
                            assert len(crop['offset']) == 2, ValueError(f"Offset for crop must be 2 values as x and y")
                            offset_x, offset_y = crop['offset'][0], crop['offset'][1]
                            case = 1
                        elif isinstance(crop['offset'], int):
                            offset_x, offset_y = crop['offset'], crop['offset']
                            case = 2
                            
                            if offset_x < obj_width:
                                offset_x = obj_width + 20
                                case = 3
                                
                            if offset_y < obj_height:
                                offset_y = obj_height + 20
                                case = 3
                            
                        else:
                            raise NotImplementedError(f"NOT yet Considered: {crop['offset']}")
                            
                            
                        if case == 1:
                            x1 = x1 - offset_x if x1 - offset_x >= 0 else 0
                            y1 = y1 - offset_y if y1 - offset_y >= 0 else 0
                            x2 = x2 + offset_x if x1 + offset_x <= width else width
                            y2 = y2 + offset_y if y1 + offset_y <= height else height
                        elif case == 2 or case == 3:
                            cx, cy = (x2 - x1)/2 + x1, (y2 - y1)/2 + y1
                            x1 = cx - offset_x/2 
                            y1 = cy - offset_y/2 
                            x2 = cx + offset_x/2 
                            y2 = cy + offset_y/2 
                            
                            if x1 < 0:
                                x2 -= x1
                                x1 = 0
                            
                            if x2 > width:
                                x1 -= (x2 - width)
                                x2 = width 
                            
                            if y1 < 0:
                                y2 -= y1
                                y1 = 0
                            
                            if y2 > height:
                                y1 -= (y2 - height)
                                y2 = height 
                            
                        _roi_img = roi_img[int(y1):int(y2), int(x1):int(x2)]
                        _roi_mask = roi_mask[int(y1):int(y2), int(x1):int(x2)]
                        
                        if case == 3:
                            _roi_img = cv2.resize(_roi_img, (int(crop['offset']), int(crop['offset'])))
                            _roi_mask = cv2.resize(_roi_mask, (int(crop['offset']), int(crop['offset'])))
                        cv2.imwrite(osp.join(image_output_dir, filename + f'_{ann_id}.{output_format}'), _roi_img)
                        cv2.imwrite(osp.join(mask_output_dir, filename + f'_{ann_id}.{output_format}'), _roi_mask)
                
                        if vis:
                            import numpy as np
                            import imgviz 
                            
                            vis_img = np.zeros((_roi_img.shape[0], _roi_img.shape[1]*2, 3)) # w, h, ch
                            vis_img[:, :_roi_img.shape[1], :] = _roi_img
                            if one_class:
                                vis_img[:, _roi_img.shape[1]:, :] = cv2.cvtColor(_roi_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
                            else:
                                color_map = imgviz.label_colormap(255)
                                vis_img[:, _roi_img.shape[1]:, :] = color_map[_roi_mask.astype(np.uint8)].astype(np.uint8) 
                            
                            cv2.imwrite(osp.join(vis_dir, filename + f'_{ann_id}.png'), vis_img)
        
            
if __name__ == '__main__':
    input_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_dataset'
    output_dir = '/HDD/datasets/projects/Tenneco/Metalbearing/outer/250110/split_mask_dataset_one_class'
    class2label = {'chamfer_mark': 1, 'line': 2, 'mark': 3}
    modes = ['train', 'val']
    # width, height = 1120, 768
    width, height = None, None
    vis = False
    output_format = 'png'
    input_format = 'bmp'
    roi = [[220, 60, 1340, 828]]
    one_class = 255
    crop = {
        'use': True, 'offset': 320 # (x, y),
    }
    
    if crop['use']:
        output_dir += '_crop'


    labelme2mask(input_dir, output_dir, class2label, input_format, modes, width, height, vis, output_format, roi, one_class, crop)

