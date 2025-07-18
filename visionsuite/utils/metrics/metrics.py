import numpy as np
from collections import Counter
import cv2
import os.path as osp


# https://github.com/rafaelpadilla/review_object_detection_metrics?tab=readme-ov-file#ap-with-iou-threshold-t05

def is_overlapped(box_1, box_2):
    if box_1[0] > box_2[2]:
        return False
    if box_2[0] > box_1[2]:
        return False
    if box_1[3] < box_2[1]:
        return False
    if box_1[1] > box_2[3]:
        return False
    
    return True

def get_area(box):
    
    area = (box[2] - box[0])*(box[3] - box[1])
    
    return area 

def get_overlap_area(box_1, box_2):
    lt_x = max(box_1[0], box_2[0])
    lt_y = max(box_1[1], box_2[1])
    rb_x = min(box_1[2], box_2[2])
    rb_y = min(box_1[3], box_2[3])

    overlap_area = (rb_x - lt_x)*(rb_y - lt_y)
    
    return overlap_area

def get_coord_diff(box_1, box_2):
    import numpy as np
    from scipy.spatial import distance

    box_1 = [box_1[i:i + 2] for i in range(0, len(box_1), 2)]
    box_2 = [box_2[i:i + 2] for i in range(0, len(box_2), 2)]

    polygon1 = np.array(box_1)
    polygon2 = np.array(box_2)

    def average_difference(poly1, poly2):
        total_diff = 0
        matched_poly2 = []

        for p1 in poly1:
            distances = distance.cdist([p1], poly2, 'euclidean')
            nearest_idx = np.argmin(distances)
            matched_poly2.append(poly2[nearest_idx])
            total_diff += distances[0, nearest_idx]

        average_diff = total_diff / len(poly1)  # 평균 차이 계산
        return average_diff

    return average_difference(polygon1, polygon2)


def get_iou(box_1, box_2, shape_type, return_dict=False):
    
    if shape_type == 'rectangle':
        if not is_overlapped(box_1, box_2):
            return 0
        
        area_1 = get_area(box_1)
        area_2 = get_area(box_2)
        
        overlap_area = get_overlap_area(box_1, box_2)
        assert overlap_area > 0, RuntimeError(f"ERROR: overlap-area must be more than 0, not {overlap_area}")
        
        
        iou = overlap_area/float(area_1 + area_2 - overlap_area)
        assert iou >= 0, RuntimeError(f"ERROR: iou must be more than 0, not {iou}")
        
        if return_dict:
            return {'iou': iou, 'area_1': area_1, 'area_2': area_2, 'overlap_area': overlap_area}
        else:
            return iou
    elif shape_type == 'polygon':
        iou, area_1, area_2, overlap_area = get_polygon_iou(box_1, box_2)
        
        if return_dict:
            return {'iou': iou, 'area_1': area_1, 'area_2': area_2, 'overlap_area': overlap_area}
        else:
            return iou        

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
    iou = intersection_area / union_area
    
    return iou, poly1.area,poly2.area, intersection_area
    
def ElevenPointInterpolatedAP(rec, prec):
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []


    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        print(r, argGreaterRecalls)

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11
    
    return [ap, rhoInterp, recallValues, None]

def mAP(result):
    ap = 0
    for r in result[1:]:
        ap += r['ap']
    mAP = ap / (len(result) - 1)
    
    return mAP

def get_average_g_results(g_results):
    result = {}
    total_avg_g_fnr, total_avg_g_fpr = 0, 0
    cnt = 0
    for image_name, g_result in g_results.items():
        avg_g_fnr, avg_g_fpr = 0, 0
        for key, val in g_result.items():
            avg_g_fnr += val['g_fnr']
            avg_g_fpr += val['g_fpr']
            cnt += 1
        total_avg_g_fnr += avg_g_fnr
        total_avg_g_fpr += avg_g_fpr
        result[image_name] = {'avg_g_fnr': avg_g_fnr/len(val), 
                              'avg_g_fpr': avg_g_fpr/len(val)
                             }
        
    result['total_avg_g_fnr'] = total_avg_g_fnr/cnt
    result['total_avg_g_fpr'] = total_avg_g_fpr/cnt
    
    return result
        
        
def update_ap_by_image(results_by_image):
    
    overall_by_class = {}
    if len(results_by_image) != 0:
        for image_name, val in results_by_image.items():
            for _class, results in val.items():
                fpr = results['fp']/(results['tp'] + results['fp'] + 1e-5)
                fnr = results['fn']/(results['tp'] + results['fn'] + 1e-5)
                results.update({'fpr': fpr, 'fnr': fnr})
                
                if _class not in overall_by_class:
                    overall_by_class[_class] = {'fpr': [], 'fnr': [], 'tp': [], 'fp': [], 'fn': [], 'tn': [], 'total_gt': [], 
                                                'miou': [], 'mean_coord_diff': []}
                    
                overall_by_class[_class]['fpr'].append(fpr)
                overall_by_class[_class]['fnr'].append(fnr)
                overall_by_class[_class]['tp'].append(results['tp'])
                overall_by_class[_class]['fp'].append(results['fp'])
                overall_by_class[_class]['fn'].append(results['fn'])
                overall_by_class[_class]['tn'].append(results['tn'])
                overall_by_class[_class]['total_gt'].append(results['total_gt'])
                overall_by_class[_class]['miou'].append(results['miou'])
                overall_by_class[_class]['mean_coord_diff'].append(results['mean_coord_diff'])
                
        for key, val in overall_by_class.items():
            overall_by_class[key]['fpr'] = np.mean(overall_by_class[key]['fpr'])
            overall_by_class[key]['fnr'] = np.mean(overall_by_class[key]['fnr'])
            overall_by_class[key]['tp'] = np.sum(overall_by_class[key]['tp'])
            overall_by_class[key]['fp'] = np.sum(overall_by_class[key]['fp'])
            overall_by_class[key]['fn'] = np.sum(overall_by_class[key]['fn'])
            overall_by_class[key]['tn'] = np.sum(overall_by_class[key]['tn'])
            overall_by_class[key]['total_gt'] = np.sum(overall_by_class[key]['total_gt'])
            overall_by_class[key]['stdiou'] = np.std(overall_by_class[key]['miou'])
            overall_by_class[key]['miou'] = np.mean(overall_by_class[key]['miou'])
            overall_by_class[key]['std_coord_diff'] = np.std(overall_by_class[key]['mean_coord_diff'])
            overall_by_class[key]['mean_coord_diff'] = np.mean(overall_by_class[key]['mean_coord_diff'])
                
    results_by_image['overall'] = overall_by_class
    
    return results_by_image
    _

def update_ious_by_image(results_by_image):
    
    if len(results_by_image) != 0:
        for image_name, val in results_by_image.items():
            for _class, results in val.items():
                results.update({'miou': np.mean(results['iou']) if len(results['iou']) != 0 else 0})
                    
    return results_by_image    

def update_coord_diff_by_image(results_by_image):
    
    if len(results_by_image) != 0:
        for image_name, val in results_by_image.items():
            for _class, results in val.items():
                results.update({'mean_coord_diff': np.mean(results['coord_diff'])})
                    
    return results_by_image  
    
def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

def get_performance(detections, ground_truths, classes, iou_threshold=0.3, method='ap', shape_type='rectangle'):
    '''
        detections: ['image filename', class-index, confidence, (x1, y1, x2, y2, ...)]
        ground_truths: ['image filename', class-index, confidence, (x1, y1, x2, y2, ...)]
    '''
    
    results_by_class = []
    results_by_image = {}
    # loop by each class for all images
    for _class in classes:
        
        # Extract info for each class
        dets = [det for det in detections if det[1] == _class]
        gts = [gt for gt in ground_truths if gt[1] == _class]
        
        num_gt = len(gts) # len(tp) + len(tn)
        
        # dets = sorted(dets, key=lambda conf: conf[2], reverse=True) # descending by confidence
        if all(conf[2] is not None for conf in dets):  
            dets = sorted(dets, key=lambda conf: conf[2], reverse=True) 

        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        gt_box_detected_map = Counter(c[0] for c in gts) # number of gt-boxes by image
        for key, val in gt_box_detected_map.items():
            gt_box_detected_map[key] = np.zeros(val)
            
        for det_index, det in enumerate(dets):
            
            # match dets and gts by image
            gt = [gt for gt in gts if gt[0] == det[0]]
            
            
            if _class == -1:
                if len(gt) == 0: ### FN
                    for ground_truth in ground_truths:
                        if ground_truth[0] == det[0]:
                            if det[0] not in results_by_image:
                                results_by_image[det[0]] = {ground_truth[1]: {'tp': 0, 'fp': 0, 'fn': 1, 'tn': 0, 'total_gt': 1, 'iou': [], 'coord_diff': []}}
                            else: 
                                if ground_truth[1] not in results_by_image[det[0]]:
                                    results_by_image[det[0]].update({ground_truth[1]: {'tp': 0, 'fp': 0, 'fn': 1, 'tn': 0, 'total_gt': 1, 'iou': [], 'coord_diff': []}})
                                else:
                                    results_by_image[det[0]][ground_truth[1]]['fn'] += 1
                                    results_by_image[det[0]][ground_truth[1]]['total_gt'] += 1
                continue
            
            if det[0] not in results_by_image:
                results_by_image[det[0]] = {_class: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_gt': len(gt), 'iou': [], 'coord_diff': []}}
            
            else: 
                if _class not in results_by_image[det[0]]:
                    results_by_image[det[0]].update({_class: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_gt': len(gt), 'iou': [], 'coord_diff': []}})

            max_iou, iou = 0, 0
            max_gt_coord, max_pred_coord = None, None
            for gt_index, _gt in enumerate(gt):
                '''
                    Within the same image, compare all gt-boxes for each det-box and then, calculate iou.
                    match the det-box and gt-box by the maximum iou.
                '''
                if len(det[3]) < 2*3:
                    continue
                iou = get_iou(det[3], _gt[3], shape_type)
                if iou > max_iou:
                    max_iou = iou 
                    max_gt_index = gt_index
                    max_gt_coord, max_pred_coord = det[3], _gt[3]
                    
                # FIXME: move into the max_iou >= iou_threshold phrase???
                if iou >= iou_threshold:
                    results_by_image[det[0]][_class]['tp'] += 1
                
            if results_by_image[det[0]][_class]['tp'] == 0:
                results_by_image[det[0]][_class]['fp'] += 1
                
            if max_iou >= iou_threshold:
                '''
                    * tp: 
                        - bigger than iou-threhold 
                        - gt-box is not matched yet
                    * fp:
                        - otherwise
                '''
                # iou
                results_by_image[det[0]][_class]['iou'].append(max_iou)
                
                # coord-diff
                if max_gt_coord is not None and  max_pred_coord is not None:
                    results_by_image[det[0]][_class]['coord_diff'].append(get_coord_diff(max_gt_coord, max_pred_coord))
                
                if gt_box_detected_map[det[0]][max_gt_index] == 0:
                    tp[det_index] = 1
                    gt_box_detected_map[det[0]][max_gt_index] = 1
                else:
                    fp[det_index] = 1
            else:
                fp[det_index] = 1
                
            # false negative
            fn = results_by_image[det[0]][_class]['total_gt'] - results_by_image[det[0]][_class]['tp']
            if fn < 0:
                results_by_image[det[0]][_class]['fn'] = 0
            else:
                results_by_image[det[0]][_class]['fn'] = fn
            
        # if len(gt_box_detected_map) != 0:
        #     if isinstance(gt_box_detected_map[det[0]] , int) and gt_box_detected_map[det[0]] != 0:
        #         results_by_image[det[0]][_class]['fn'] = len(gt_box_detected_map[det[0]]) - np.sum(gt_box_detected_map[det[0]])
        #     elif isinstance(gt_box_detected_map[det[0]] , np.ndarray):
        #         results_by_image[det[0]][_class]['fn'] = len(gt_box_detected_map[det[0]]) - np.sum(gt_box_detected_map[det[0]])
                    
        accumulated_tp = np.cumsum(tp)
        accumulated_fp = np.cumsum(fp)
        accumulated_precision = np.divide(accumulated_tp, (accumulated_tp + accumulated_fp))
        accumulated_recall = accumulated_tp/num_gt if num_gt != 0 else accumulated_tp
                    
        if method.lower() == 'ap':
            [ap, mean_precision, mean_recall, ii] = calculateAveragePrecision(accumulated_recall, accumulated_precision)
        else:
            [ap, mean_precision, mean_recall, _] = ElevenPointInterpolatedAP(accumulated_recall, accumulated_precision)

        result_by_class = {
            'class' : _class,
            'accumulated_precision' : accumulated_precision,
            'accumulated_recall' : accumulated_recall,
            'precision': accumulated_precision[-1] if len(accumulated_precision) != 0 else 0,
            'recall': accumulated_recall[-1] if len(accumulated_recall) != 0 else 0,
            'ap' : ap,
            'interpolated_precision' : mean_precision,
            'interpolated_recall' : mean_recall,
            'total_gt' : num_gt,
            'total_tp' : np.sum(tp),
            'total_fp' : np.sum(fp),
            'total_fn' : num_gt - np.sum(tp),
        }
        
        results_by_class.append(result_by_class)


    for image_name, results in results_by_image.items():
        for _class in classes[1:]:
            gts = [gt for gt in ground_truths if gt[1] == _class]
            gt = [gt for gt in gts if gt[0] == image_name]

            if _class not in results:
                results.update({_class: {'tp': 0, 'fp': 0, 'fn': len(gt), 'tn': 0, 'total_gt': len(gt), 'iou': [], 'coord_diff': []}})
        

    results_by_image = update_ious_by_image(results_by_image)
    results_by_image = update_coord_diff_by_image(results_by_image)
    results_by_image = update_ap_by_image(results_by_image)
    results_by_class.append({'map': mAP(results_by_class)})
    
    
        
    
        
    return {'by_class': results_by_class, 'by_image': dict(sorted(results_by_image.items()))}

