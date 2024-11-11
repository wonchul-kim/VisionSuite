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
        self.run_callbacks('on_build_loop_start')

        self.trainer = build_trainer(**self.args['train']['trainer'])()
        self.trainer.build(model=self.model, loss=self.loss, optimizer=self.optimizer, 
                           lr_scheduler=self.lr_scheduler, dataloader=self.train_dataloader, 
                           scaler=self.scaler, archive=self.archive,
                           **self.args['train'])
        self.validator = build_validator(**self.args['val']['validator'])()
        self.validator.build(model=self.model, loss=self.loss, dataloader=self.val_dataloader,
                             label2index=self.dataset.label2index, 
                             device=self.args['train']['device'], topk=self.args['train']['topk'],
                             archive=self.archive, **self.args['val'])
        
        self.run_callbacks('on_build_loop_end')
        
    def run_loop(self):
        super().run_loop()
        self.run_callbacks('on_run_loop_start')
        for epoch in range(self.start_epoch, self.args['train']['epochs']):

            if self.args['distributed']['use']:
                self.dataset.train_sampler.set_epoch(epoch)
            self.trainer.train(epoch)
            self.validator.val(epoch)
            
        self.run_callbacks('on_run_loop_end')