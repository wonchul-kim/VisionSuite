import yaml
import os.path as osp
from visionsuite.engines.utils.helpers import create_output_dir, mkdir
from visionsuite.engines.utils.loggers.monitor import Monitor
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule


class Archive(BaseOOPModule):
    def __init__(self, output_dir=None):
        self.args = None
        self._output_dir = output_dir
        
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self._output_dir = create_output_dir(self.args['output_dir'], make_dirs=True)
        assert osp.exists(self._output_dir), RuntimeError(f"Output dir ({self._output_dir}) has not been created")
        
        self.create_directories()
        self._load()
        
    def _load(self):
        self._load_monitor()
        
    def _load_monitor(self):
        if self.args['monitor']['use']:
            self.monitor = Monitor()
            self.monitor.set(output_dir=self.logs_dir, fn='monitor')
        else:
            self.monitor = None
            
    @property
    def output_dir(self):
        return self._output_dir
    
    def save_args(self, args):
        with open(osp.join(self.cfgs_dir, 'args.yaml'), 'w') as yf:
            yaml.dump(args, yf, default_flow_style=False)
            
    def create_directories(self):
        for directory in ['val', 'cfgs', 'logs', 'weights']:
            mkdir(osp.join(self._output_dir, directory))
            setattr(self, directory + '_dir', osp.join(self._output_dir, directory))    
    
            
        
        
            
    