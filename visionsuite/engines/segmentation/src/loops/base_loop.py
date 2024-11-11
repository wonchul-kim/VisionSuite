import os.path as osp
from abc import abstractmethod
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule
from visionsuite.engines.segmentation.src.dataloaders.build import build_dataloader
from visionsuite.engines.segmentation.src.losses.build import build_loss
from visionsuite.engines.segmentation.src.optimizers.build import build_optimizer
from visionsuite.engines.segmentation.src.schedulers.build import build_scheduler
from visionsuite.engines.utils.torch_utils.utils import load_ckpt
from visionsuite.engines.segmentation.utils.registry import LOOPS

class BaseLoop(BaseOOPModule):
    
    required_attributes = ['model', 'train_dataloader', 'lr_scheduler', 'loss', 'optimizer', 'dataset',
                           'trainer', 'start_epoch']
    
    def __init__(self):
        super().__init__()
        self.args = None 
        
        self.loop = None
        
        self.model = None 
        self.train_dataloader = None 
        self.val_dataloader = None 
        self.lr_scheduler = None 
        self.loss = None 
        self.optimizer = None
        self.scaler = None 
        
        self.dataset = None
        self.archive = None
        
        self._start_epoch = None
        self.trainer = None 
        self.validator = None
        
    @property 
    def start_epoch(self):
        return self._start_epoch 
    
    @start_epoch.setter 
    def start_epoch(self, val):
        self._start_epoch = val

    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self.model = _model
        self.dataset = _dataset
        self.archive = _archive
        
        self.train_dataloader = build_dataloader(dataset=self.dataset, mode='train', 
                                                 **self.args['dataloader'], augment=self.args['augment'])
        self.val_dataloader = build_dataloader(dataset=self.dataset, mode='val', 
                                                 **self.args['dataloader'], augment=self.args['augment'])
        
        self.loss = build_loss(**self.args['loss'])
        self.optimizer = build_optimizer(model=self.model, **self.args['optimizer'])
        self.lr_scheduler = build_scheduler(optimizer=self.optimizer, epochs=self.args['train']['epochs'], **self.args['scheduler'])
        
        self._set_resume()
        self.loop = LOOPS.get(self.args['loop']['type'])

    @BaseOOPModule.track_status
    @abstractmethod
    def run_loop(self):
        pass
    
    def _set_resume(self):
        if self.args['resume']['use'] and self.args['train']['ckpt']:
            assert osp.exists(self.args['train']['ckpt']), ValueError(f"There is no such checkpoint: {self.args['train']['ckpt']}")
            ckpt = load_ckpt(self.args['train']['ckpt'])
            
            self.model.model_without_ddp.load_state_dict(ckpt['model'], strict=True)
            
            attributes = ['optimizer', 'lr_scheduler', 'epoch']
            if self.args['train']['amp']:
                attributes.append('scaler')
            
            for attribute in attributes:
                if self.args['resume'][attribute]:
                    if attribute in ckpt:
                        if attribute == 'epoch':
                            getattr(self, 'start_epoch').load_state_dict(ckpt[attribute])
                        else:
                            getattr(self, attribute).load_state_dict(ckpt[attribute])
                    else:
                        raise AttributeError(f"There is no {attribute} in ckpt({self.args['train']['ckpt']})")
                    
        else:
            self._start_epoch = 1
            