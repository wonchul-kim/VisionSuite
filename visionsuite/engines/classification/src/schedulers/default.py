import torch
from visionsuite.engines.classification.utils.registry import SCHEDULERS


@SCHEDULERS.register()
def get_scheduler(optimizer, epochs, scheduler_config):
    if scheduler_config['scheduler_name'] == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                            step_size=scheduler_config['lr_step_size'], 
                                                            gamma=scheduler_config['lr_gamma'])
    elif scheduler_config['scheduler_name'] == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs - scheduler_config['lr_warmup_epochs'], eta_min=scheduler_config['lr_min']
        )
    elif scheduler_config['scheduler_name'] == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                                   gamma=scheduler_config['lr_gamma'])
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{scheduler_config['scheduler_name']}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if scheduler_config['lr_warmup_epochs'] > 0:
        if scheduler_config['lr_warmup_method'] == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=scheduler_config['lr_warmup_decay'], total_iters=scheduler_config['lr_warmup_epochs']
            )
        elif scheduler_config['lr_warmup_method'] == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=scheduler_config['lr_warmup_decay'], total_iters=scheduler_config['lr_warmup_epochs']
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{scheduler_config['lr_warmup_method']}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[scheduler_config['lr_warmup_epochs']]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
        
    return lr_scheduler