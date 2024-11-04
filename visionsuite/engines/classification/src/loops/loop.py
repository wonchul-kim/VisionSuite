import torch 
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule
from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.classification.src.dataloaders.build import build_dataloader
from visionsuite.engines.classification.src.losses.build import build_loss
from visionsuite.engines.classification.src.optimizers.build import build_optimizer
from visionsuite.engines.classification.src.schedulers.build import build_scheduler
from visionsuite.engines.classification.utils.registry import LOOPS


class Loop(BaseOOPModule):
    def __init__(self):
        self.args = None 
        
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
    def run(self, model, dataset, archive=None, callbacks=None):
        train_dataloader = build_dataloader(self.args, dataset, mode='train')
        val_dataloader = build_dataloader(self.args, dataset, mode='val')
        
        loss = build_loss(self.args['loss'])
        optimizer = build_optimizer(model.model, self.args['optimizer'])

        scaler = torch.cuda.amp.GradScaler() if self.args['amp'] else None
        lr_scheduler = build_scheduler(optimizer, self.args['epochs'], self.args['scheduler'], self.args['warmup_scheduler'])
        self.args['start_epoch'] = set_resume(self.args['resume'], self.args['ckpt'], model.model_without_ddp, 
                                    optimizer, lr_scheduler, scaler, self.args['amp'])
        
        loop = LOOPS.get(self.args['loop']['type'])
        loop(callbacks, self.args, dataset.train_sampler, 
                        model.model, model.model_without_ddp, loss, optimizer, 
                        train_dataloader, model.model_ema, scaler, archive,
                        lr_scheduler, val_dataloader, dataset.label2index)