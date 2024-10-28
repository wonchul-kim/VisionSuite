import os
import os.path as osp

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.utils.torch_utils.utils import save_on_master
from visionsuite.engines.utils.archives import BaseArchive
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.callbacks import callbacks as cls_callbacks

from visionsuite.engines.classification.utils.transforms import get_mixup_cutmix
from visionsuite.engines.classification.losses.default import get_cross_entropy_loss
from visionsuite.engines.classification.schedulers.default import get_scheduler
from visionsuite.engines.classification.dataloaders.default import get_dataloader
from visionsuite.engines.classification.optimizers.default import get_optimizer
from visionsuite.engines.classification.models.default import get_model, get_ema_model
from visionsuite.engines.classification.train.default import train_one_epoch
from visionsuite.engines.classification.val.default import evaluate
from visionsuite.engines.classification.datasets.default import load_data
from visionsuite.engines.utils.torch_utils.utils import set_weight_decay
from visionsuite.engines.classification.pipelines.variables import set_variables




def main(args):
    
    set_variables(args)
    
    archive = BaseArchive(osp.join(args.output_dir, 'classification'), monitor=True)
    archive.save_args(args)
    
    callbacks = Callbacks(_callbacks=cls_callbacks)
    
    
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    classes = dataset.classes
    print(f"Classes: {classes}")
    label2class = {label: _class for label, _class in enumerate(classes)}
    
    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader, data_loader_test = get_dataloader(dataset, dataset_test, train_sampler, test_sampler, args.batch_size, args.workers, collate_fn)
    
    model, model_without_ddp = get_model(args.model, args.device, num_classes, args.distributed, args.sync_bn, args.weights)

    criterion = get_cross_entropy_loss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    optimizer = get_optimizer(args.opt, args.lr, parameters, args.momentum, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = get_scheduler(optimizer, args.lr_scheduler, args.lr_step_size, args.lr_gamma, args.epochs, 
                  args.lr_warmup_method, args.lr_warmup_epochs, args.lr_min, args.lr_warmup_decay)
    model_ema = get_ema_model(model_without_ddp, args.device, args.model_ema, args.world_size, args.batch_size, args.model_ema_steps, args.model_ema_decay, args.epochs)
    args.start_epoch = set_resume(args.resume, args.ckpt, model_without_ddp, 
                                  optimizer, lr_scheduler, scaler, args.amp)
    
    callbacks.run_callbacks('on_train_start')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, 
                        args.device, epoch, args, callbacks, model_ema, scaler, 
                        args.topk, archive)
        lr_scheduler.step()
        
        if archive.weights_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            save_on_master(checkpoint, os.path.join(archive.weights_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(archive.weights_dir, "checkpoint.pth"))

        callbacks.run_callbacks('on_val_start')
        evaluate(model_ema if model_ema else model, criterion, data_loader_test, args.device, epoch, label2class, callbacks, 
                 topk=args.topk, log_suffix="EMA" if args.model_ema else "", archive=archive)
        callbacks.run_callbacks('on_val_end')
        
    callbacks.run_callbacks('on_train_end')
    
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
