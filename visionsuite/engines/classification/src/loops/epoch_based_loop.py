import os.path as osp

from visionsuite.engines.utils.torch_utils.utils import save_on_master

from visionsuite.engines.classification.src.trainers.build import build_trainer
from visionsuite.engines.classification.src.validators.build import build_validator
from visionsuite.engines.classification.utils.registry import LOOPS
from visionsuite.engines.classification.src.loops.base_loop import BaseLoop
from visionsuite.engines.classification.utils.results import TrainResults, ValResults


@LOOPS.register()
class EpochBasedLoop(BaseLoop):
    def __init__(self):
        super().__init__()
        
    def build(self, _model, _dataset, _callbacks=None, _archive=None, *args, **kwargs):
        super().build(_model, _dataset, _callbacks=_callbacks, _archive=_archive, *args, **kwargs)
        
        self.train_results = TrainResults()
        self.val_results = ValResults()
        
        self.trainer = build_trainer(**self.args['train']['trainer'])(
                                self.model.model, self.loss, self.optimizer, self.train_dataloader, 
                                self.args['train']['device'], self.args, self.callbacks, self.model.model_ema, self.scaler, 
                                self.args['train']['topk'], self.archive, self.train_results)
        self.validator = build_validator(**self.args['val']['validator'])
        
        
    def run(self):
        super().run()
        self.callbacks.run_callbacks('on_train_start')
        for epoch in range(self.args['start_epoch'], self.args['train']['epochs']):
            if self.args['distributed']['use']:
                self.dataset.train_sampler.set_epoch(epoch)
            self.trainer.run(epoch)
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

            self.callbacks.run_callbacks('on_val_start')
            self.validator(self.args['val'], self.model.model_ema if self.model.model_ema else self.model.model, 
                     self.loss, self.val_dataloader, self.args['train']['device'], epoch, 
                     self.dataset.label2index, self.callbacks, 
                    topk=self.args['train']['topk'], log_suffix="EMA" if self.args['model']['ema']['use'] else "", 
                    archive=self.archive, results=self.val_results)
            self.callbacks.run_callbacks('on_val_end')
            
        self.callbacks.run_callbacks('on_train_end')
        