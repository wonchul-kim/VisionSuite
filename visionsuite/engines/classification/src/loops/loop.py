import torch 
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule
from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.classification.src.dataloaders.build import build_dataloader
from visionsuite.engines.classification.utils.registry import (LOSSES, OPTIMIZERS, SCHEDULERS, LOOPS)

class Loop(BaseOOPModule):
    def __init__(self):
        
        self.args = None 
        
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
        print(args, kwargs)
        
    def run(self, model, dataset, archive=None, callbacks=None):
        train_dataloader = build_dataloader(self.args, dataset, mode='train')
        val_dataloader = build_dataloader(self.args, dataset, mode='val')
        
        loss = LOSSES.get("loss")(self.args['loss'])
        optimizer = OPTIMIZERS.get('optimizer')(model.model, self.args['optimizer'])

        scaler = torch.cuda.amp.GradScaler() if self.args['amp'] else None
        lr_scheduler = SCHEDULERS.get('lr_scheduler')(optimizer, self.args['epochs'], self.args['scheduler'], self.args['warmup_scheduler'])
        self.args['start_epoch'] = set_resume(self.args['resume'], self.args['ckpt'], model.model_without_ddp, 
                                    optimizer, lr_scheduler, scaler, self.args['amp'])
        
        
        loop = LOOPS.get(self.args['loop']['type'])
        loop(callbacks, self.args, dataset.train_sampler, 
                        model.model, model.model_without_ddp, loss, optimizer, 
                        train_dataloader, model.model_ema, scaler, archive,
                        lr_scheduler, val_dataloader, dataset.label2index)