from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

from abc import abstractmethod
import os.path as osp
import argparse 
import yaml

from visionsuite.engines.utils.helpers import yaml2namespace
from visionsuite.engines.utils.torch_utils.utils import parse_device_ids, set_torch_deterministic, get_device
from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode


class BaseTrainRunner:
    def __init__(self, task):
        self._task = task
        
        ## define basic atrributes to train
        self._archive = None
        self._callbacks = None 
        self._model = None 
        self._datasets = {'train': None, 'val': None}
        self._dataloaders = {'train': None, 'val': None}
        self._scheduler = None 
        self._optimizer = None 
        self._loop = None 
        self._loss = None
        
    @property 
    def task(self):
        return self._task 
        
    @abstractmethod
    def set_configs(self, default_cfgs_file=None, *args, **kwargs):
        
        def _parse_args():
            self.args.device_ids = parse_device_ids(self.args.device_ids)
            
        if default_cfgs_file is None:
            default_cfgs_file=ROOT.parents[1] / self._task / 'cfgs/default.yaml'
        with open(default_cfgs_file, 'r') as yf:
            default_cfgs = yaml.load(yf)
            
        default_cfgs.update(vars(self.args))
        
        for key in kwargs.keys():
            assert key in default_cfgs, ValueError(f"There is no such key({key}) in configs")
        
        default_cfgs.update(kwargs)
        
        self.args = argparse.Namespace(**default_cfgs)
        _parse_args()

    @abstractmethod
    def set_variables(self):
        init_distributed_mode(self.args)
        set_torch_deterministic(self.args.use_deterministic_algorithms)
        self.args.device = get_device(self.args.device)
        
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
        self.set_variables()
        self.set_dataset()
        self.run_loop()
    