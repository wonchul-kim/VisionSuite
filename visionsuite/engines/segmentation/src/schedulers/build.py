from torch.optim.lr_scheduler import PolynomialLR
import torch 

def build_scheduler(**config):
    

def get_scheduler(args, optimizer, iters_per_epoch):
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args['epochs'] - args['lr_warmup_epochs']), power=0.9
    )

    if args['lr_warmup_epochs'] > 0:
        warmup_iters = iters_per_epoch * args['lr_warmup_epochs']
        args['lr_warmup_method'] = args['lr_warmup_method'].lower()
        if args['lr_warmup_method'] == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args['lr_warmup_decay'], total_iters=warmup_iters
            )
        elif args['lr_warmup_method'] == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args['lr_warmup_decay'], total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args['lr_warmup_method']}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler