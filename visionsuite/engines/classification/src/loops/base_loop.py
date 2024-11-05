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
    def __init__(self):
        super().__init__()
        self.args = None 
        
        self.model = None 
        self.train_dataloader = None 
        self.val_dataloader = None 
        self.lr_scheduler = None 
        self.scale = None 
        self.loss = None 
        self.optimizer = None
        
        self.dataset = None
        self.callbacks = None
        self.archive = None
            
    def build(self, _model, _dataset, _callbacks=None, _archive=None, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self.model = _model
        self.dataset = _dataset
        self.callbacks = _callbacks
        self.archive = _archive
        
        self.train_dataloader = build_dataloader(self.args, self.dataset, mode='train')
        self.val_dataloader = build_dataloader(self.args, self.dataset, mode='val')
        
        self.loss = build_loss(self.args['loss'])
        self.optimizer = build_optimizer(self.model.model, self.args['optimizer'])

        self.scaler = torch.cuda.amp.GradScaler() if self.args['train']['amp'] else None
        self.lr_scheduler = build_scheduler(self.optimizer, self.args['train']['epochs'], 
                                            self.args['scheduler'], self.args['warmup_scheduler'])
        self.args['start_epoch'] = set_resume(self.args['resume']['use'], self.args['train']['ckpt'], self.model.model_without_ddp, 
                                    self.optimizer, self.lr_scheduler, self.scaler, self.args['train']['amp'])
        
        self.loop = LOOPS.get(self.args['loop']['type'])

    @BaseOOPModule.track_status
    @abstractmethod
    def run(self):
        pass