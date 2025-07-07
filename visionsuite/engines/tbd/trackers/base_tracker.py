import yaml 
import os.path as osp
from abc import *

from ultralytics import YOLO 
from bytetrack.byte_tracker import BYTETracker
from tracktrack.tracker import TrackTracker

class BaseTracker:
    def __init__(self, config_file):

        assert osp.exists(config_file), ValueError(f"There is no such config-file at {config_file}")        
        with open(config_file) as yf:
            self._config = yaml.load(yf, Loader=yaml.FullLoader)

        self._set()
        
    @property
    def config(self):
        return self._config 
    
    def _set(self):
        self.set_detector()
        self.set_tracker()
        
    def set_detector(self):
        self._detector = YOLO(self._config['detector']['model_name']) 
    
    def set_tracker(self):
        self._tracker = BYTETracker(**self._config['tracker'])
            
    @abstractmethod    
    def track(self, image):
        pass