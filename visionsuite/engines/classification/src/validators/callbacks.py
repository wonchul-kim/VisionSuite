import time 
import os.path as osp
from visionsuite.engines.utils.helpers import mkdir
from visionsuite.engines.utils.functionals import denormalize
from visionsuite.engines.classification.utils.vis.vis_val import save_validation

def on_val_epoch_start(validator, *args, **kwargs):
    pass

def on_val_epoch_end(validator, *args, **kwargs):
    validator.metric_logger.synchronize_between_processes()

    def _save_val():
        if validator.archive.args['save_val']['use'] and osp.exists(validator.archive.val_dir):
            vis_dir = osp.join(validator.archive.val_dir, str(kwargs['epoch']))
            if not osp.exists(vis_dir):
                mkdir(vis_dir)
                
            save_validation(validator.model, validator.dataloader, validator.label2class, kwargs['epoch'], vis_dir, validator.device, denormalize)

    def _save_results():
        validator.results.epoch = int(kwargs['epoch'])
        validator.results.loss = float(round(validator.metric_logger.meters['loss'].global_avg, 4))
        validator.results.accuracy = float(round(validator.metric_logger.meters["acc1"].global_avg, 4))
        validator.results.time_for_a_epoch = float(round(time.time() - kwargs['start_time_epoch'], 3))

    _save_val()
    _save_results()


def on_val_batch_start(validator, *args, **kwargs):
    pass

def on_val_batch_end(validator, *args, **kwargs):
    pass 

def on_val_step_start(validator, *args, **kwargs): # iteration for a batch
    pass

def on_val_step_end(validator, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_val_epoch_start": [on_val_epoch_start],
    "on_val_epoch_end": [on_val_epoch_end],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_step_start": [on_val_step_start],
    "on_val_step_end": [on_val_step_end],
}