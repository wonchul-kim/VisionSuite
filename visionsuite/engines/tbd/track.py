import os 
import os.path as osp 
from glob import glob 
from ultralytics import YOLO 
import cv2
import numpy as np
import time
import yaml

from trackers.bytetrack.byte_tracker import BYTETracker
from trackers.tracktrack.tracker import TrackTracker

        
if __name__ == '__main__':
    
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    output_dir = '/HDD/etc/outputs/tracking/bytetrack/yolo'
    os.makedirs(output_dir, exist_ok=True)

    input_dir = '/HDD/datasets/public/MOT17/test/MOT17-01-DPM/img1'
    image_format = 'jpg'
    num_frames = 300

    model_name = 'yolo11n.pt'
    DET_CONFIDENCE = 0.3
    DET_IOU = 0.5
    model = YOLO(model_name)

    img_files = sorted(glob(osp.join(input_dir, f'*.{image_format}')))

    seq_info = open(osp.join(input_dir, '../seqinfo.ini'), mode='r')
    for s_i in seq_info.readlines():
        if 'frameRate' in s_i:
            max_time_lost = int(s_i.split('=')[-1]) * 2
        if 'imWidth' in s_i:
            img_w = int(s_i.split('=')[-1])
        if 'imHeight' in s_i:
            img_h = int(s_i.split('=')[-1])

    config_file = '/HDD/_projects/github/VisionSuite/visionsuite/engines/tbd/trackers/configs/bytetrack.yaml'
    # config_file = '/HDD/_projects/github/VisionSuite/visionsuite/engines/tbd/trackers/configs/tracktrack.yaml'
    with open(config_file) as yf:
        config = yaml.load(yf, Loader=yaml.FullLoader)
        
    img_size = (img_h, img_w)
    tracker = BYTETracker(**config['tracker'])
    # import argparse
    # tracker = TrackTracker(argparse.Namespace(**config['tracker']))

    total_time, total_count = 0, 0
    start = time.time()

    for img_file in img_files[:num_frames]:
        filename = osp.split(osp.splitext(img_file)[0])[-1]  
        img = cv2.imread(img_file)
        dets = model(img_file, verbose=False, conf=DET_CONFIDENCE, iou=DET_IOU)[0]
        
        boxes = dets.boxes
        low_detections, high_detections = [], []
        detections = []
        for idx, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
            detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item()])

        if len(detections) != 0:
            tracked_outputs = tracker.track(np.array(detections), (img_h, img_w), (img.shape[0], img.shape[1]),
                                            aspect_ratio_thresh=config['post']['aspect_ratio_thresh'], 
                                            min_box_area=config['post']['min_box_area'])

            for tlwh, id, score in zip(*tracked_outputs):
                cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                            (0, 0, 255), 2)
                cv2.putText(img, f"{id}_{score:0.1f}", 
                            (int(tlwh[0]), int(tlwh[1] - 10)), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2,
                            color=(0, 0, 255), lineType=3
                        )
        
        cv2.imwrite(osp.join(output_dir, filename + '.jpg'), img)
            