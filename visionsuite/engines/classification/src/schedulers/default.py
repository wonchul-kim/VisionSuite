import torch

def get_scheduler(optimizer, lr_scheduler, lr_step_size, lr_gamma, epochs, 
                  lr_warmup_method, lr_warmup_epochs, lr_min, lr_warmup_decay):
    
    if lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs - lr_warmup_epochs, eta_min=lr_min
        )
    elif lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
        
    return lr_scheduler