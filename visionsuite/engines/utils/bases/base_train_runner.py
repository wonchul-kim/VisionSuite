from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

from abc import abstractmethod
import os.path as osp

from visionsuite.engines.utils.helpers import yaml2dict, update_dict
from visionsuite.engines.utils.torch_utils.utils import parse_device_ids, set_torch_deterministic, get_device
from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode
from visionsuite.engines.classification.utils.params import TrainParams
from visionsuite.engines.utils.archives import Archive

            
class BaseTrainRunner:
    def __init__(self, task):
        super().__init__()
        
        self._task = task
        self.pipelines = {}
        
    @property 
    def task(self):
        return self._task 
        
    @abstractmethod
    def set_configs(self, cfgs_file=None, default_cfgs_file=None, *args, **kwargs):

        assert osp.exists(cfgs_file), ValueError(f'There is no such cfgs file: {cfgs_file}')

        cfgs = yaml2dict(cfgs_file)

        def _parse_args(args):
            args['train']['device_ids'] = parse_device_ids(args['train']['device_ids'])
            
        if default_cfgs_file is None:
            default_cfgs_file=ROOT.parents[1] / self._task / 'cfgs/default.yaml'
        default_cfgs = yaml2dict(default_cfgs_file)

        update_dict(default_cfgs, cfgs)  
        update_dict(default_cfgs, kwargs)          
        
        self.args = default_cfgs
        _parse_args(self.args)

    @abstractmethod
    def set_variables(self):
        init_distributed_mode(self.args)
        set_torch_deterministic(self.args['train']['use_deterministic_algorithms'])
        self.args['train']['device'] = get_device(self.args['train']['device'])
        
        self._archive = Archive(osp.join(self.args['archive']['output_dir'], self._task), monitor=True)
        self._archive.save_args(self.args)
        
    @abstractmethod
    def run(self):
        pass 
    
    
    def train(self, cfgs_file=None, *args, **kwargs):
        
        self.set_configs(cfgs_file=cfgs_file, *args, **kwargs)
        self.set_variables()
        self.run()
    