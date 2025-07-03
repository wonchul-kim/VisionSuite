import os 
import os.path as osp 
from glob import glob 
from ultralytics import YOLO 
import cv2
import numpy as np
import time

from trackers.byte_tracker import BYTETracker

        
if __name__ == '__main__':
    
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    output_dir = '/HDD/etc/outputs/tracking/bytetrack/yolo'
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
    # args.min_box_area = 100
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
    args.track_thresh = 0.2
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.aspect_ratio_thresh = 1.6
    args.min_box_area = 10
    args.mot20 = False 
    
    seq_info = open(osp.join(input_dir, '../seqinfo.ini'), mode='r')
    for s_i in seq_info.readlines():
        if 'frameRate' in s_i:
            args.max_time_lost = int(s_i.split('=')[-1]) * 2
        if 'imWidth' in s_i:
            img_w = int(s_i.split('=')[-1])
        if 'imHeight' in s_i:
            img_h = int(s_i.split('=')[-1])

    img_size = (img_h, img_w)
    tracker = BYTETracker(args)

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
            detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])

        if len(detections) != 0:
            online_targets = tracker.update(np.array(detections), (img_h, img_w), (img.shape[0], img.shape[1]))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)


            for tlwh, id, score in zip(online_tlwhs, online_ids, online_scores):
                cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                            (0, 0, 255), 2)
                # cv2.putText(img, f"{id}_{names[int(cls)]}_{conf:0.2f}", 
                # cv2.putText(img, f"{id}_{names[int(cls)]}", 
                cv2.putText(img, f"{id}_{score:0.1f}", 
                            (int(tlwh[0]), int(tlwh[1] - 10)), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(0, 0, 255), lineType=3
                        )
        
        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), img)
            