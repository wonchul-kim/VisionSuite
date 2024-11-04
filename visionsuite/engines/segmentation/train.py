import os

import torch
from datetime import datetime
from visionsuite.engines.segmentation.train.default import train_one_epoch
from visionsuite.engines.segmentation.val.default import evaluate
from visionsuite.engines.segmentation.optimizers.default import get_optimizer
from visionsuite.engines.segmentation.schedulers.default import get_scheduler
from visionsuite.engines.segmentation.models.default import get_model
from visionsuite.engines.segmentation.datasets.default import get_dataset
from visionsuite.engines.segmentation.dataloaders.default import get_dataloader
from visionsuite.engines.segmentation.losses.default import criterion
from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode
from visionsuite.engines.utils.torch_utils.utils import set_torch_deterministic, get_device, save_on_master, parse_device_ids
from visionsuite.engines.utils.helpers import mkdir
from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.utils.loggers.monitor import Monitor

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(x, mean=MEAN, std=STD):
    x *= np.array(std)
    x += np.array(mean)
    x = x.astype(np.uint8)
    
    return x


def main(args):
    import os.path as osp 
    
    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

    args.device_ids = parse_device_ids(args.device_ids)

    now = datetime.now()
    hour = now.hour
    minute = now.minute
    second = now.second
    
    args.output_dir = osp.join(args.output_dir, f'_{args.model}_{hour}_{minute}_{second}')
    
    if args.output_dir:
        mkdir(args.output_dir, True)

    init_distributed_mode(args)
    device = get_device(args.device)
    set_torch_deterministic(args.use_deterministic_algorithms)

    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)
    data_loader, data_loader_test, train_sampler, test_sampler = get_dataloader(args, dataset, dataset_test)
    model, model_without_ddp, params_to_optimize = get_model(args, num_classes, device)
    
    optimizer = get_optimizer(args, params_to_optimize)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    iters_per_epoch = len(data_loader)
    lr_scheduler = get_scheduler(args, optimizer, iters_per_epoch)

    args.start_epoch = set_resume(args.resume, args.ckpt, model_without_ddp, 
                                  optimizer, lr_scheduler, scaler, args.amp)
    
    monitor = Monitor()
    monitor.set(output_dir=args.output_dir, fn='monitor')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_metric_logger = train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat, val_metric_logger = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        
        from utils.vis.vis_val import save_validation
        vis_dir = osp.join(args.output_dir, f'vis/{epoch}')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        
        save_validation(model, device, dataset_test, 4, epoch, vis_dir, denormalize)
        print(confmat)
        
        monitor.log({"learning rate": train_metric_logger.meters['lr'].value})
        monitor.log({"train avg loss": train_metric_logger.meters['loss'].avg})
        for key, val in confmat.values.items():
            if 'acc' in key:
                monitor.log({key: val})
            if 'iou' in key:
                monitor.log({key: val})
        monitor.save()

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            'scale': scaler.state_dict() if scaler and args.amp else None
        }
        save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

def get_args_parser():
    import argparse
    import yaml 
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent
    
    with open(ROOT / "cfgs/default.yaml", 'r') as yf:
        cfgs = yaml.safe_load(yf)
        
    args = argparse.Namespace(**cfgs)
    
    return args


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
