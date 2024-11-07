import os.path as osp

from visionsuite.engines.utils.torch_utils.utils import save_on_master

from visionsuite.engines.classification.src.trainers.build import build_trainer
from visionsuite.engines.classification.src.validators.build import build_validator
from visionsuite.engines.classification.utils.registry import LOOPS
from visionsuite.engines.classification.src.loops.base_loop import BaseLoop
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import callbacks

@LOOPS.register()
class EpochBasedLoop(BaseLoop, Callbacks):
    def __init__(self):
        BaseLoop.__init__(self)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)

    def build(self, _model, _dataset, _archive=None, *args, **kwargs):
        super().build(_model, _dataset, _archive=_archive, *args, **kwargs)
        self.run_callbacks('on_build_start')

        self.trainer = build_trainer(**self.args['train']['trainer'])()
        self.trainer.build(self.model.model, self.loss, self.optimizer, self.train_dataloader, 
                            self.args['train']['device'], self.args, self.model.model_ema, self.scaler, 
                            self.args['train']['topk'], self.archive)
        self.validator = build_validator(**self.args['val']['validator'])()
        self.validator.build(self.args['val'], self.model.model_ema if self.model.model_ema else self.model.model, 
                     self.loss, self.val_dataloader, self.args['train']['device'],  
                     self.dataset.label2index, 
                    topk=self.args['train']['topk'], log_suffix="EMA" if self.args['model']['ema']['use'] else "", 
                    archive=self.archive)
        
        self.run_callbacks('on_build_end')
        
    def run_loop(self):
        super().run_loop()
        self.run_callbacks('on_run_loop_start')
        for epoch in range(self.current_epoch, self.args['train']['epochs']):

            if self.args['distributed']['use']:
                self.dataset.train_sampler.set_epoch(epoch)
            self.trainer.train(epoch)
            self.lr_scheduler.step()

            #TODO: MOVE THIS INTO CALLBACK AND ADD BEST ----------------------------------------------------
            if self.archive.weights_dir:
                checkpoint = {
                    "model": self.model.model_without_ddp.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": self.args,
                }
                if self.model.model_ema:
                    checkpoint["model_ema"] = self.model.model_ema.state_dict()
                if self.scaler:
                    checkpoint["scaler"] = self.scaler.state_dict()
                
                if self.archive.args['save_model']['last']:
                    save_on_master(checkpoint, osp.join(self.archive.weights_dir, "last.pth"))    
                    
                if epoch%self.archive.args['save_model']['freq_epoch'] == 0:
                    save_on_master(checkpoint, osp.join(self.archive.weights_dir, f"model_{epoch}.pth"))
            # ----------------------------------------------------------------------------------------------

            self.validator.val(epoch)
            
        self.run_callbacks('on_run_loop_end')