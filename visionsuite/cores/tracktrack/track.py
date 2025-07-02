import os 
import os.path as osp 
from glob import glob 
from ultralytics import YOLO 
import cv2
import numpy as np
from trackers.tracker import Tracker
import time
        
if __name__ == '__main__':
    
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    output_dir = '/HDD/etc/outputs/tracking/tracktrack/yolo'
    os.makedirs(output_dir, exist_ok=True)

    model_name = 'yolo11n.pt'
    input_dir = '/HDD/datasets/public/MOT17/test/MOT17-01-DPM/img1'
    image_format = 'jpg'
    det_conf = 0.3
    det_iou = 0.5
    num_frames = 10

    model = YOLO(model_name)

    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))

    import argparse
    args = argparse.ArgumentParser("Tracker").parse_args()
    
    args.min_len = 3
    args.min_box_area = 100
    args.max_time_lost = 30
    args.penalty_p = 0.20
    args.penalty_q = 0.40
    args.reduce_step = 0.05
    args.tai_thr = 0.55
    # args.det_thr, args.init_thr = 0.65, 0.75
    # args.det_thr, args.init_thr = 0.60, 0.60
    # args.det_thr, args.init_thr = 0.45, 0.55
    args.det_thr, args.init_thr = 0.20, 0.20
    args.match_thr = 0.70
    
    
    seq_info = open(osp.join(input_dir, '../seqinfo.ini'), mode='r')
    for s_i in seq_info.readlines():
        if 'frameRate' in s_i:
            args.max_time_lost = int(s_i.split('=')[-1]) * 2
        if 'imWidth' in s_i:
            args.img_w = int(s_i.split('=')[-1])
        if 'imHeight' in s_i:
            args.img_h = int(s_i.split('=')[-1])
    tracker = Tracker(args, 'MOT17-01-DPM')

    total_time, total_count = 0, 0
    start = time.time()

    for img_file in img_files[:num_frames]:
        filename = osp.split(osp.splitext(img_file)[0])[-1]  
        img = cv2.imread(img_file)
        dets = model(img_file, verbose=False, conf=det_conf, iou=det_iou)[0]
        
        boxes = dets.boxes
        low_detections, high_detections = [], []
        detections = []
        for idx, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
            # if conf >= 0.3:
            #     low_detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), cls.item(), conf.item()])
            # elif conf >= 0.3:
            #     high_detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), cls.item(), conf.item()])
            low_detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])
        high_detections = low_detections
        # if len(low_detections) != 0 and len(high_detections) != 0:        
        #     track_results = tracker.update(np.array(low_detections), np.array(high_detections))
        if len(low_detections) != 0:        
            track_results = tracker.update(np.array(low_detections), np.array(low_detections))
        else:
            track_results = tracker.update_without_detections()
        
        total_time += time.time() - start
        total_count += 1
          
        # Filter out the results
        x1y1whs, track_ids, scores = [], [], []
        for t in track_results:
            # Check aspect ratio
            if t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                continue

            # Check track id, minimum box area
            if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                x1y1whs.append(t.x1y1wh)
                track_ids.append(t.track_id)
                scores.append(t.score)
                
            cv2.rectangle(img, (int(t.x1y1x2y2[0]), int(t.x1y1x2y2[1])), (int(t.x1y1x2y2[2]), int(t.x1y1x2y2[3])),
                        (0, 0, 255), 2)
            # cv2.putText(img, f"{id}_{names[int(cls)]}_{conf:0.2f}", 
            # cv2.putText(img, f"{id}_{names[int(cls)]}", 
            cv2.putText(img, f"{t.track_id}_{t.score:0.2f}", 
                        (int(t.x1y1x2y2[0]), int(t.x1y1x2y2[1] - 10)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 0, 255), lineType=3
                    )
        
        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), img)
        