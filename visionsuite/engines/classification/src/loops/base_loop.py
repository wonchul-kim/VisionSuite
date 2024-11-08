import torch 
from abc import abstractmethod
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule
from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.classification.src.dataloaders.build import build_dataloader
from visionsuite.engines.classification.src.losses.build import build_loss
from visionsuite.engines.classification.src.optimizers.build import build_optimizer
from visionsuite.engines.classification.src.schedulers.build import build_scheduler
from visionsuite.engines.classification.utils.registry import LOOPS

class BaseLoop(BaseOOPModule):
    
    required_attributes = ['model', 'train_dataloader', 'lr_scheduler', 'loss', 'optimizer', 'dataset',
                           'trainer', 'current_epoch']
    
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
        
        self.dataset = None
        self.archive = None
        
        self._current_epoch = None
        self.trainer = None 
        self.validator = None
        
    @property 
    def current_epoch(self):
        return self._current_epoch 
    
    @current_epoch.setter 
    def current_epoch(self, val):
        self._current_epoch = val
            
    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self.model = _model
        self.dataset = _dataset
        self.archive = _archive
        
        self.train_dataloader = build_dataloader(self.args, self.dataset, mode='train')
        self.val_dataloader = build_dataloader(self.args, self.dataset, mode='val')
        
        self.loss = build_loss(self.args['loss'])
        self.optimizer = build_optimizer(self.model.model, self.args['optimizer'])
        self.lr_scheduler = build_scheduler(self.optimizer, self.args['train']['epochs'], 
                                            self.args['scheduler'], self.args['warmup_scheduler'])
        self._current_epoch = 1
        # set_resume(self.args['resume']['use'], self.args['train']['ckpt'], self.model.model_without_ddp, 
        #                             self.optimizer, self.lr_scheduler, self.scaler, self.args['train']['amp'])
        
        self.loop = LOOPS.get(self.args['loop']['type'])

    @BaseOOPModule.track_status
    @abstractmethod
    def run_loop(self):
        pass