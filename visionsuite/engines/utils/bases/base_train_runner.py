from abc import abstractmethod
import os.path as osp
import argparse 
import yaml
from visionsuite.engines.utils.helpers import yaml2namespace
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent


class BaseTrainRunner:
    def __init__(self, task):
        self._task = task

    @property 
    def task(self):
        return self._task 
        
    @abstractmethod
    def set_configs(self, default_cfgs_file=None, *args, **kwargs):
        if default_cfgs_file is None:
            default_cfgs_file=ROOT.parents[1] / self._task / 'cfgs/default.yaml'
        with open(default_cfgs_file, 'r') as yf:
            default_cfgs = yaml.load(yf)
            
        default_cfgs.update(vars(self.args))
        
        for key in kwargs.keys():
            assert key in default_cfgs, ValueError(f"There is no such key({key}) in configs")
        
        default_cfgs.update(kwargs)
        
        self.args = argparse.Namespace(**default_cfgs)
    
    @abstractmethod
    def set_dataset(self):
        pass 
    
    def set_model(self):
        pass

    @abstractmethod
    def run_loop(self):
        pass 
    
    def train(self, cfgs_file, *args, **kwargs):
        
        assert osp.exists(cfgs_file), ValueError(f'There is no such cfgs file: {cfgs_file}')
        
        self.args = yaml2namespace(cfgs_file)
        self.set_configs(*args, **kwargs)
        self.set_dataset()
        self.run_loop()
    