import torch
from visionsuite.engines.classification.utils.registry import SCHEDULERS


@SCHEDULERS.register()
def lr_scheduler(optimizer, epochs, scheduler_config, warmup_scheduler_config):
    from visionsuite.engines.utils.helpers import get_params_from_obj
    if scheduler_config['type'] == "CosineAnnealingLR":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs - scheduler_config['lr_warmup_epochs'], eta_min=scheduler_config['lr_min']
        )
    else:
        scheduler_obj = SCHEDULERS.get(scheduler_config['type'])
        if not scheduler_obj:
            raise RuntimeError(
                f"Invalid lr scheduler '{scheduler_config['type']}'")
        scheduler_params = get_params_from_obj(scheduler_obj)
        for key in scheduler_params.keys():
            if key in scheduler_config:
                scheduler_params[key] = scheduler_config[key]
                
            if key == 'optimizer':
                scheduler_params[key] = optimizer
                
                
        main_lr_scheduler = scheduler_obj(**scheduler_params)
    
    if warmup_scheduler_config and warmup_scheduler_config['total_iters'] > 0:
        scheduler_obj = SCHEDULERS.get(warmup_scheduler_config['warmup_type'])
        if not scheduler_obj:
            raise RuntimeError(
                f"Invalid lr scheduler '{warmup_scheduler_config['warmup_type']}'. Only ConstantLR and LinearLR are available")
        scheduler_params = get_params_from_obj(scheduler_obj)
        for key in scheduler_params.keys():
            if key in warmup_scheduler_config:
                scheduler_params[key] = warmup_scheduler_config[key]
                
            if key == 'optimizer':
                scheduler_params[key] = optimizer
                
        warmup_lr_scheduler = scheduler_obj(**scheduler_params)
        lr_scheduler = SCHEDULERS.get(warmup_scheduler_config['integrate_type'])(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_scheduler_config['total_iters']]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler