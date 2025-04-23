import os 
import os.path as osp 
from glob import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
from shapely.geometry import Polygon
import pandas as pd


def polygon2rect(points, offset=0):
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    
    return [[min(xs) - offset, min(ys) - offset], [max(xs) + offset, min(ys) - offset], 
            [max(xs) + offset, max(ys) + offset], [min(xs) - offset, max(ys) + offset]]
    
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

def polygon_area(points):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    # x와 y를 한 칸씩 시프트하여 곱셈
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def get_blobs(image, contour_thres=3):
    """ return
    
        { 'blobs': [
                    { "polygon": [[], [], ...], "bbox": [[], [], ...] },
                    { "polygon": [[], [], ...], "bbox": [[], [], ...] }, 
                    ...
                ], 
        'mean area': 10,
        'min area': 1,
        'max area': 20,
        'std area': 2,
    }
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    blobs = []
    areas = []
    for contour in contours:
        if len(contour) < contour_thres:
            pass
        else:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for objects in [approx]:
                polygon = []
                xs, ys = [], []
                for jdx in range(0, len(objects)):
                    polygon.append(
                        [
                            int(objects[jdx][0][0]),
                            int(objects[jdx][0][1]),
                        ]
                    )
                    xs.append(int(objects[jdx][0][0]))
                    ys.append(int(objects[jdx][0][1]))
                    
                bbox = [[min(xs), min(ys)], [max(xs), max(ys)]]
                blobs.append({'polygon': polygon, 'bbox': bbox})
                areas.append(polygon_area(polygon))

    return blobs, areas


def vis_raw_output_by_channel(trt_arr, python_arr, _output_dir, filename):
    ### Vis. raw output 
    channels = trt_arr.shape[-1]
    vmin = min(trt_arr.min(), python_arr.min())
    vmax = max(trt_arr.max(), python_arr.max())

    fig, axes = plt.subplots(3, channels, figsize=(5 * channels, 10))
    mappable = None

    for c in range(channels):
        ax = axes[0, c]
        im = ax.imshow(trt_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"TRT (Channel {c})")
        ax.axis("off")
        if mappable is None:
            mappable = im

    for c in range(channels):
        ax = axes[1, c]
        ax.imshow(python_arr[..., c], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"Python (Channel {c})")
        ax.axis("off")

    for c in range(channels):
        ax = axes[2, c]
        ax.imshow(abs(python_arr[..., c] - trt_arr[..., c]), cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"ABS(Python - TRT) (Channel {c})")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # 오른쪽에 세로로 길게
    fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, cax=cbar_ax)

    plt.suptitle("Raw Prediction per Channel", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])  # [left, bottom, right, top]
    plt.savefig(osp.join(_output_dir, filename + '_raw.png'))
    plt.close()

def vis_pred(trt_input_dir, python_input_dir, _output_dir, filename):
    # pred images ------------------------------------------------------------
    trt_pred_img = cv2.imread(osp.join(trt_input_dir, filename + '.bmp.bmp.png'))
    if trt_pred_img is None:
        trt_pred_img = cv2.imread(osp.join(trt_input_dir, filename + '.bmp.bmp'))
    python_pred_img = cv2.imread(osp.join(python_input_dir, '../' + filename + '.png'))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(trt_pred_img, cmap='gray')
    axes[0].axis("off")

    axes[1].imshow(python_pred_img, cmap='gray')
    axes[1].axis("off")

    plt.suptitle(f"trt (left) vs. python (right)", fontsize=14)
    plt.tight_layout()
    plt.savefig(osp.join(_output_dir, filename + '_pred.png'))
    plt.close()
    
def vis_historgram_by_channel(trt_arr, python_arr, is_same, rtol, atol, _output_dir, filename):
    channels = trt_arr.shape[-1]
    # --- 히스토그램 ---
    # First histogram
    fig = plt.figure(figsize=(30, 10))
    plt.subplots_adjust(bottom=0.2)  # 하단 여백을 충분히 줌
    fig.text(0.5, 0.04, f"same = {is_same}",  ha='center', fontsize=10)
    fig.text(0.5, 0.01, f"(rtol = {rtol}, atol = {atol}", ha='center', fontsize=10)
    for channel in range(channels):
        plt.subplot(2, channels, channel + 1)
        n, bins, patches = plt.hist(trt_arr[..., channel].flatten(), bins=100, color='skyblue', edgecolor='k')
        plt.title(f"TRT channel {channel}")
        plt.xlabel("Value")
        plt.ylabel("Pixel Count(log)")
        plt.grid(True)
        plt.yscale("log")  # 로그 스케일

        # # 막대 위에 개수 표시
        # for count, bin_left, patch in zip(n, bins, patches):
        #     if count > 0:
        #         plt.text(bin_left + (bins[1] - bins[0]) / 2, count, f"{int(count)}", 
        #                 ha='center', va='bottom', fontsize=6, rotation=90)

    # Second histogram
    for channel in range(channels):
        plt.subplot(2, channels, channel + 1 + channels)
        n2, bins2, patches2 = plt.hist(python_arr[..., channel].flatten(), bins=100, color='skyblue', edgecolor='k')
        plt.title(f"Python channel {channel}")
        plt.xlabel("Value")
        plt.ylabel("Pixel Count(log)")
        plt.grid(True)
        plt.yscale("log")  # 로그 스케일
        
        # for count, bin_left, patch in zip(n2, bins2, patches2):
        #     if count > 0:
        #         plt.text(bin_left + (bins2[1] - bins2[0]) / 2, count, f"{int(count)}", 
        #                 ha='center', va='bottom', fontsize=6, rotation=90)

    # plt.tight_layout()
    plt.savefig(osp.join(_output_dir, filename + '_histo.jpg'))
    plt.close()


def main():
    height, width, num_classes = 768, 1120, 4
    rtol = 1e-2
    atol = 1e-2    
    contour_threshold = 10
    contour_confidence_threshold = 0.1
    case_iou_threshold = 0.1
    case_offset = 30

    num_same, num_not_same = 0, 0
    trt_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/trt_vs_python/trt/1'       
    python_input_dir = f'/DeepLearning/etc/_athena_tests/benchmark/trt_vs_python/test/exp/vis/raw'
    output_dir = f'/DeepLearning/etc/_athena_tests/benchmark/trt_vs_python/compare/rtol{rtol}_atol{atol}'

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    bin_files = glob(osp.join(trt_input_dir, '*.bin'))
    # bin_files = glob(osp.join(trt_input_dir, '125020717055621_14_Outer_1.bmp.bmp.bin'))

    result = {}
    '''
        result = { "image filename": {
                                        "trt": { "channel 1": { 'blobs': [
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] },
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] }, 
                                                                            ...
                                                                        ], 
                                                                'mean area': 10,
                                                                'min area': 1,
                                                                'max area': 20,
                                                                'std area': 2,
                                                            }, 
                                                 "channel 2": { 'blobs': [
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] },
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] }, 
                                                                            ...
                                                                        ], 
                                                                'mean area': 10,
                                                                'min area': 1,
                                                                'max area': 20,
                                                                'std area': 2,
                                                            }, 
                                        "python": { "channel 1": { 'blobs': [
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] },
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] }, 
                                                                            ...
                                                                        ], 
                                                                'mean area': 10,
                                                                'min area': 1,
                                                                'max area': 20,
                                                                'std area': 2,
                                                            }, 
                                                 "channel 2": { 'blobs': [
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] },
                                                                            { "polygon": [[], [], ...], "bbox": [[], [], ...] }, 
                                                                            ...
                                                                        ], 
                                                                'mean area': 10,
                                                                'min area': 1,
                                                                'max area': 20,
                                                                'std area': 2,
                                                            }, 
                                                }
                                        "case": 0,
                                    },
                                    ...
                }
        }
    '''
    idx = 0
    for bin_file in tqdm(bin_files):
        idx += 1
        filename = osp.split(osp.splitext(bin_file)[0])[-1].split(".")[0]
        npz_file = osp.join(python_input_dir, filename + '.npz')
        
        if not osp.exists(npz_file):
            print(f'There is no such npz file: {npz_file}')
            continue
        
        
        trt_arr = np.fromfile(bin_file, dtype=np.float32).reshape((num_classes, height, width))
        trt_arr = np.transpose(trt_arr, (1, 2, 0))
        python_arr = np.load(npz_file)
        if 'arr' in python_arr:
            python_arr = np.load(npz_file)['arr']
        if python_arr.shape == (num_classes, height, width):
            python_arr = np.transpose(python_arr, (1, 2, 0))
            
        
        assert trt_arr.shape == python_arr.shape # hwc
        assert trt_arr.dtype == python_arr.dtype # float32    
                
        is_close = np.allclose(trt_arr, python_arr, rtol=rtol, atol=atol)
        is_same = is_close

        _output_dir = osp.join(output_dir, str(is_same))
        if not osp.exists(_output_dir):
            os.mkdir(_output_dir)
        
        # vis_raw_output_by_channel(trt_arr, python_arr, _output_dir, filename)
        # vis_pred(trt_input_dir, python_input_dir, _output_dir, filename)
        # vis_historgram_by_channel(trt_arr, python_arr, is_same, rtol, atol, _output_dir, filename)

        case = -1
        trt_blobs, python_blobs = {}, {}
        for channel in range(1, trt_arr.shape[-1]):
            trt_arr[trt_arr < contour_confidence_threshold] = 0
            python_arr[python_arr < contour_confidence_threshold] = 0
            
            image = trt_arr[..., channel].astype(np.uint8)
            blobs, areas = get_blobs(image, contour_thres=contour_threshold)
            if areas != []:
                trt_blobs[channel] = {'blobs': blobs, 'mean area': np.mean(areas), 'min area': np.min(areas), 'max area': np.max(areas), 'std area': np.std(areas), 'count': len(blobs)}
            else:
                trt_blobs[channel] = {'blobs': blobs, 'mean area': None, 'min area': None, 'max area': None, 'std area': None}
            image = python_arr[..., channel].astype(np.uint8)
            blobs, areas = get_blobs(image, contour_thres=contour_threshold)
            if areas != []:
                python_blobs[channel] = {'blobs': blobs, 'mean area': np.mean(areas), 'min area': np.min(areas), 'max area': np.max(areas), 'std area': np.std(areas), 'count': len(blobs)}
            else:
                python_blobs[channel] = {'blobs': blobs, 'mean area': None, 'min area': None, 'max area': None, 'std area': None}


            if len(trt_blobs[channel]['blobs']) == 0 and len(python_blobs[channel]['blobs']) == 0:
                case = 1
            elif (len(trt_blobs[channel]['blobs']) == 0 and len(python_blobs[channel]['blobs']) != 0) or (len(trt_blobs[channel]['blobs']) != 0 and len(python_blobs[channel]['blobs']) == 0):
                case = 3
            else:
                
                for trt_blob in trt_blobs[channel]['blobs']:
                    for python_blob in python_blobs[channel]['blobs']:
                        
                        iou, area1, area2, intersection_area = get_polygon_iou(tuple(coord for points in polygon2rect(trt_blob['polygon']) for coord in points), 
                                              tuple(coord for points in polygon2rect(python_blob['polygon']) for coord in points))
                        if iou > case_iou_threshold:
                            case = 2
                        else:
                            case = 3
                            break

        result[filename] = {'trt': trt_blobs, 'python': python_blobs, 'case': case}
        
        if idx > 3:
            break
    
    with open(osp.join(output_dir, 'result.json'), 'w') as jf:
        json.dump(result, jf, indent=4)
        
    with open(osp.join(output_dir, 'result.pkl'), "wb") as f:
        pickle.dump(result, f)



    rows = []

    for img_name, img_data in result.items():
        for method in ['trt', 'python']:
            for channel in range(1, num_classes):
                blob_data = img_data[method][channel]
                row = {
                    'image': img_name,
                    'method': method,
                    'channel': channel,
                    'mean_area': blob_data['mean area'],
                    'min_area': blob_data['min area'],
                    'max_area': blob_data['max area'],
                    'std_area': blob_data['std area'],
                    'case': img_data['case'],
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(osp.join(output_dir, "results.csv"), index=False)

    num_txt = open(osp.join(output_dir, 'numbers.txt'), 'w')
    num_txt.write(f'number of same: {num_same}\n')
    num_txt.write(f'number of NOT same: {num_not_same}\n')
    num_txt.close()


if __name__ == '__main__':
    main()