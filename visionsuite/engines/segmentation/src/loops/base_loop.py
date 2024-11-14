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
    
    def __init__(self, name=None):
        super().__init__(name=name)
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
        self.log_info(f"BUILT train_dataloader: {self.train_dataloader}", self.build.__name__, __class__.__name__)
        
        self.val_dataloader = build_dataloader(dataset=self.dataset, mode='val', 
                                                 **self.args['dataloader'], augment=self.args['augment'])
        self.log_info(f"BUILT val_dataloader: {self.val_dataloader}", self.build.__name__, __class__.__name__)        
        
        self.loss = build_loss(**self.args['loss'])
        self.log_info(f"BUILT loss: {self.loss}", self.build.__name__, __class__.__name__)        
        
        self.optimizer = build_optimizer(model=self.model, **self.args['optimizer'])
        self.log_info(f"BUILT optimizer: {self.optimizer}", self.build.__name__, __class__.__name__)        
        
        self.lr_scheduler = build_scheduler(optimizer=self.optimizer, 
                                            epochs=self.args['train']['epochs'], 
                                            iters_per_epoch=len(self.train_dataloader),
                                            **self.args['scheduler'])
        self.log_info(f"BUILT lr_scheduler: {self.lr_scheduler}", self.build.__name__, __class__.__name__)        
        
        self._set_resume()
        self.loop = LOOPS.get(self.args['loop']['type'])

    @BaseOOPModule.track_status
    @abstractmethod
    def run_loop(self):
        self.log_info(f"START loop", self._set_resume.__name__, __class__.__name__)  
    
    def _set_resume(self):
        if self.args['resume']['use']:
            assert osp.exists(self.args['resume']['seed_model']), ValueError(f"There is no such checkpoint: {self.args['resume']['seed_model']}")
            ckpt = load_ckpt(self.args['resume']['seed_model'])
            self.log_info(f"LOADED ckpt: {ckpt.keys()}", self._set_resume.__name__, __class__.__name__)
            
            self.model.model_without_ddp.load_state_dict(ckpt['model'], strict=True)
            self.log_info(f"LOADED seed_model weights", self._set_resume.__name__, __class__.__name__)
            
            attributes = ['optimizer', 'lr_scheduler', 'epoch']
            if self.args['train']['amp']:
                attributes.append('scaler')
            
            for attribute in attributes:
                if self.args['resume'][attribute]:
                    if attribute in ckpt:
                        if attribute == 'epoch':
                            self._start_epoch = int(ckpt[attribute]) + 1
                            
                            if self.args['train']['epochs'] - 10 <= self.start_epoch <= self.args['train']['epochs'] + 10:
                                self.args['train']['epochs'] = self.start_epoch + 100
                                
                                self.log_info(f"Epochs is updated to {self.start_epoch + 100} since the loaded epoch is {int(ckpt[attribute])}", self._set_resume.__name__, __class__.__name__)
                        else:
                            getattr(self, attribute).load_state_dict(ckpt[attribute])
                            self.log_info(f"LOADED {attribute}", self._set_resume.__name__, __class__.__name__)
                    else:
                        raise AttributeError(f"There is no {attribute} in ckpt({self.args['train']['ckpt']})")
            
            self.log_info(f"SET resume lr_scheduler: start_epoch is {self._start_epoch}", self._set_resume.__name__, __class__.__name__)       
        else:
            self._start_epoch = 1
            self.log_info(f"NO resume lr_scheduler: start_epoch is {self._start_epoch}", self._set_resume.__name__, __class__.__name__)     