import os.path as osp 
import json
from glob import glob
from tqdm import tqdm
import cv2
import os
import numpy as np

def ond_dim_points_to_polygon(points):
    # points를 (x, y) 형식의 튜플 리스트로 변환
    return [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

def get_polygon_iou(point1, point2):
    from shapely.geometry import Polygon
    # 두 다각형을 Polygon 객체로 변환
    poly1 = Polygon(ond_dim_points_to_polygon(point1))
    poly2 = Polygon(ond_dim_points_to_polygon(point2))
    
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
        
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
        
    # 교집합 영역 계산
    intersection_area = poly1.intersection(poly2).area
    
    
    union_area = poly1.union(poly2).area
    
    # IoU 계산
    iou = intersection_area / (union_area + 0.000001)
    
    return iou, poly1.area,poly2.area, intersection_area

def compare_two(anns1, anns2):
    is_different = False
    for ann1 in anns1:
                
        label1 = ann1['label']
        points1 = ann1['points']
        
        if len(points1) < 3:
            continue
            
        for ann2 in anns2:
            
            label2 = ann2['label']
            points2 = ann2['points']
            
            
            if label1 != label2:
                is_different = True 
                break
            
            if len(points2) < 3:
                continue
            
            iou, area1, area2, intersection_area = get_polygon_iou(tuple(coord for points in points1 for coord in points), 
                                                                   tuple(coord for points in points2 for coord in points))

            if iou < iou_threshold:
                is_different = True 
                break 
            
        if is_different:
            break 

    return is_different

base_dir = '/HDD/etc/repeatablility'
dir_names = ['gcnet_epochs100', 'mask2former_epochs140', 'pidnet_l_epochs300']
iou_threshold = 0.25
for dir_name in dir_names:
    json_files1 = sorted(glob(osp.join(base_dir, dir_name, 'test/exp/labels', '*.json')))

    no_diff_no_points = 0
    no_diff_points = 0
    no_diff_points_files = []
    diff_points = 0
    
    for json_file1 in tqdm(json_files1, desc=dir_name):
        filename = osp.split(osp.splitext(json_file1)[0])[-1]
        json_file2 = osp.join(base_dir, dir_name, 'test/exp2/labels', f'{filename}.json')
        json_file3 = osp.join(base_dir, dir_name, 'test/exp3/labels', f'{filename}.json')
        
        assert osp.exists(json_file2), ValueError(f"There is no such file: {json_file2}")
        assert osp.exists(json_file3), ValueError(f"There is no such file: {json_file3}")
        
        with open(json_file1, 'r') as jf:
            anns1 = json.load(jf)['shapes']
        
        with open(json_file1, 'r') as jf:
            anns2 = json.load(jf)['shapes']
        
        with open(json_file1, 'r') as jf:
            anns3 = json.load(jf)['shapes']
        
        is_different = False
        
        if len(anns1) != len(anns2) or len(anns1) != len(anns3) or len(anns2) != len(anns3):
            is_different = True 
        else:
            if len(anns1) == 0 and len(anns2) == 0 and len(anns3) == 0:
                no_diff_no_points += 1
                is_different = False
            elif compare_two(anns1, anns2) or compare_two(anns1, anns3) or compare_two(anns2, anns1) or compare_two(anns2, anns3) or compare_two(anns3, anns1) or compare_two(anns3, anns2):
                is_different = True 
                diff_points += 1
            else:
                is_different = False
                no_diff_points += 1
                no_diff_points_files.append(filename)
                        
        if is_different:
            img1 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp/vis/{filename}_3_0.png'))
            img2 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp2/vis/{filename}_3_0.png'))
            img3 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp3/vis/{filename}_3_0.png'))
            
            diff_dir = osp.join(base_dir, dir_name, 'diff_w_points')
            if not osp.exists(diff_dir):
                os.mkdir(diff_dir)
            
            cv2.imwrite(osp.join(diff_dir, f'{filename}.png'), np.vstack([img1, img2, img3]))
        
                        
        if filename in no_diff_points_files:
            img1 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp/vis/{filename}_3_0.png'))
            img2 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp2/vis/{filename}_3_0.png'))
            img3 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp3/vis/{filename}_3_0.png'))
            
            no_diff_dir = osp.join(base_dir, dir_name, 'no_diff_w_points')
            if not osp.exists(no_diff_dir):
                os.mkdir(no_diff_dir)
            
            cv2.imwrite(osp.join(no_diff_dir, f'{filename}.png'), np.vstack([img1, img2, img3]))
        
            
    assert len(json_files1) == no_diff_no_points + no_diff_points + diff_points
            
    txt = open(osp.join(base_dir, dir_name, 'diff.txt'), 'w')
    txt.write(f"No diff. b/c no points: {no_diff_no_points}\n")
    txt.write(f"No diff. even with points: {no_diff_points}\n")
    txt.write(f"diff. with points: {diff_points}\n")
    txt.close()
        
            
            
                
        
                        
            
                
                    
                    
               
                
    
    


        
    
        
    
    
    