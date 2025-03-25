import os.path as osp 
import json
from glob import glob
from tqdm import tqdm
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def ond_dim_points_to_polygon(points):
    # points를 (x, y) 형식의 튜플 리스트로 변환
    return [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

def get_polygon_iou(point1, point2):
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

def polygon2rect(points, offset=0):
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    
    return [[min(xs) - offset, min(ys) - offset], [max(xs) + offset, min(ys) - offset], 
            [max(xs) + offset, max(ys) + offset], [min(xs) - offset, max(ys) + offset]]

def compare_two(anns1, anns2, iou_threshold, area_threshold=0, rect_iou=False, offset=0):
    
    is_different = False
    
    if len(anns1) == 0:
        return True 
    
    for ann1 in anns1:
                
        label1 = ann1['label']
        points1 = ann1['points']
        
        if rect_iou:
            points1 = polygon2rect(points1, offset=offset)
        
        if len(points1) < 3:
            continue
            
        if len(anns2) == 0:
            is_different = True 
            break
            
        for ann2 in anns2:
            
            label2 = ann2['label']
            points2 = ann2['points']
            if rect_iou:
                points1 = polygon2rect(points2, offset=offset)

            
            if label1 != label2:
                is_different = True 
                break
            
            if len(points2) < 3:
                continue
            
            iou, area1, area2, intersection_area = get_polygon_iou(tuple(coord for points in points1 for coord in points), 
                                                                   tuple(coord for points in points2 for coord in points))

            if area_threshold and (area1 < area_threshold and area2 < area_threshold):
                continue

            if iou < iou_threshold:
                is_different = True 
                break 
            
        if is_different:
            break 
        else:
            poly1 = Polygon(ond_dim_points_to_polygon(tuple(coord for points in points1 for coord in points)))
            
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            
            if poly1.area < area_threshold:
                is_different = False 

    return is_different

def run(base_dir, dir_names, iou_thresholds, area_thresholds, vis=False, figs=True, rect_iou=False, offset=0):
       
    for dir_name in dir_names:
        results = {}
        for iou_threshold in iou_thresholds:
            for area_threshold in area_thresholds:
                json_files1 = sorted(glob(osp.join(base_dir, dir_name, 'test/exp/labels', '*.json')))

                no_diff_no_points = 0
                no_diff_points = 0
                no_diff_points_files = []
                diff_points = 0
                
                for json_file1 in tqdm(json_files1, desc=f"{dir_name}: IoU({iou_threshold}) & Area({area_threshold})"):
                    filename = osp.split(osp.splitext(json_file1)[0])[-1]
                    
                    # if filename == '125020717054728_7_Outer_1_Image': 
                    if filename == '125020717054826_4_Outer_1_Image':
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    
                    json_file2 = osp.join(base_dir, dir_name, 'test/exp2/labels', f'{filename}.json')
                    json_file3 = osp.join(base_dir, dir_name, 'test/exp3/labels', f'{filename}.json')
                    
                    assert osp.exists(json_file2), ValueError(f"There is no such file: {json_file2}")
                    assert osp.exists(json_file3), ValueError(f"There is no such file: {json_file3}")
                    
                    with open(json_file1, 'r') as jf:
                        anns1 = json.load(jf)['shapes']
                    
                    with open(json_file2, 'r') as jf:
                        anns2 = json.load(jf)['shapes']
                    
                    with open(json_file3, 'r') as jf:
                        anns3 = json.load(jf)['shapes']
                    
                    is_different = False
                    if len(anns1) == 0 and len(anns2) == 0 and len(anns3) == 0:
                        # all cases are no points
                        no_diff_no_points += 1
                        is_different = False
                    elif compare_two(anns1, anns2, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset) or compare_two(anns1, anns3, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset) or compare_two(anns2, anns1, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset) or compare_two(anns2, anns3, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset) or compare_two(anns3, anns1, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset) or compare_two(anns3, anns2, iou_threshold, area_threshold, rect_iou=rect_iou, offset=offset):
                        # different predicted points
                        is_different = True 
                        diff_points += 1
                    else:
                        # no difference even with predicted points
                        is_different = False
                        no_diff_points += 1
                        no_diff_points_files.append(filename)
                    
                    if vis and is_different:
                        img1 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp/vis/{filename}_3_0.png'))
                        img2 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp2/vis/{filename}_3_0.png'))
                        img3 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp3/vis/{filename}_3_0.png'))
                        
                        diff_dir = osp.join(base_dir, dir_name, f'iou{iou_threshold}_area{area_threshold}', 'diff_w_points')
                        if not osp.exists(diff_dir):
                            os.makedirs(diff_dir)
                        
                        cv2.imwrite(osp.join(diff_dir, f'{filename}.png'), np.vstack([img1, img2, img3]))
                    
                                    
                    if vis and filename in no_diff_points_files:
                        img1 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp/vis/{filename}_3_0.png'))
                        img2 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp2/vis/{filename}_3_0.png'))
                        img3 = cv2.imread(osp.join(base_dir, dir_name, f'test/exp3/vis/{filename}_3_0.png'))
                        
                        no_diff_dir = osp.join(base_dir, dir_name,  f'iou{iou_threshold}_area{area_threshold}', 'no_diff_w_points')
                        if not osp.exists(no_diff_dir):
                            os.makedirs(no_diff_dir)
                        
                        cv2.imwrite(osp.join(no_diff_dir, f'{filename}.png'), np.vstack([img1, img2, img3]))
                    
                        
                assert len(json_files1) == no_diff_no_points + no_diff_points + diff_points, ValueError(f"{len(json_files1)} == {no_diff_no_points} + {no_diff_points} + {diff_points}")
                        
                results[f'iou{iou_threshold}_area{area_threshold}'] = {'no_diff_no_points': no_diff_no_points, 'no_diff_points': no_diff_points, 'diff_points': diff_points}
                
                if figs:
                    txt = open(osp.join(base_dir, dir_name, 'diff.txt'), 'a')
                    txt.write(f"\n >>>>>>>>>>>>>>>> IoU Threshold: {iou_threshold} & Area Threshold: {area_threshold} <<<<<<<<<<<<<<<<\n")
                    txt.write(f"No diff. b/c no points: {no_diff_no_points}\n")
                    txt.write(f"No diff. even with points: {no_diff_points}\n")
                    txt.write(f"diff. with points: {diff_points}\n")
                    txt.write("============================================================================\n")
                    txt.close()

        if figs:
            df = pd.DataFrame.from_dict(results)     
            df.to_csv(osp.join(base_dir, dir_name, 'diff.csv'))
            
            # X축 레이블
            categories = list(results.keys())

            # Y축 값 분리
            no_diff_no_points = [results[key]['no_diff_no_points'] for key in categories]
            no_diff_points = [results[key]['no_diff_points'] for key in categories]
            diff_points = [results[key]['diff_points'] for key in categories]

            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

            # 라인 그래프 설정
            colors = ['blue', 'orange', 'green']
            labels = ['no_diff_no_points', 'no_diff_points', 'diff_points']
            data = [no_diff_no_points, no_diff_points, diff_points]

            # 각 서브플롯에 데이터 추가
            for i, ax in enumerate(axes):
                ax.plot(categories, data[i], marker='o', color=colors[i], linestyle='-', label=labels[i])
                ax.set_ylabel('Count')
                ax.set_title(labels[i])
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 데이터 포인트 값 표시
                for j, txt in enumerate(data[i]):
                    ax.text(j, txt, str(txt), ha='center', va='bottom', fontsize=10)

            # X축 설정
            axes[-1].set_xticks(np.arange(len(categories)))
            axes[-1].set_xticklabels(categories, rotation=45, ha='right')

            # 전체 제목 추가
            fig.suptitle('IoU & Area Settings - Line Plot Comparison', fontsize=14)
            plt.xlabel('IoU & Area Settings')

            # 그래프 출력
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(osp.join(base_dir, dir_name, 'diff.png'))
            
if __name__ == '__main__':

    base_dir = '/HDD/etc/repeatablility'
    dir_names = ['gcnet_epochs100', 'mask2former_epochs140', 'pidnet_l_epochs300', 'sam2_epochs300']
    rect_iou = True 
    offset = 10
    
    ### ===================================
    iou_thresholds = [0.05, 0.1, 0.2, 0.3]
    area_thresholds = [10, 50, 100, 150, 200]
    figs = True 
    vis = False
    run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset)
    ### ===================================
    iou_thresholds = [0.05]
    area_thresholds = [100]
    figs = False
    vis = True
    run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset)
                   
    
    # base_dir = '/HDD/etc/repeatablility'
    # dir_names = ['sam2_epochs300']
    
    # iou_thresholds = [0.05]
    # area_thresholds = [100]
    # figs = True
    # vis = True
    # rect_iou = True 
    # offset = 10
    # run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset)
                   
    
                            
            
                
                    
                    
               
                
    
    


        
    
        
    
    
    