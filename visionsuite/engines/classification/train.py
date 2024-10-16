import datetime
import os
import time
import os.path as osp
import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
from visionsuite.engines.utils.loggers.monitor import Monitor
from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.utils.torch_utils.dist import init_distributed_mode
from visionsuite.engines.utils.helpers import mkdir, get_cache_path
from visionsuite.engines.utils.torch_utils.utils import set_torch_deterministic, get_device, save_on_master, parse_device_ids
from visionsuite.engines.classification.losses.default import get_cross_entropy_loss
from visionsuite.engines.classification.schedulers.default import get_scheduler
from visionsuite.engines.classification.dataloaders.default import get_dataloader
from visionsuite.engines.classification.optimizers.default import get_optimizer
from visionsuite.engines.classification.models.default import get_model, get_ema_model
from visionsuite.engines.classification.train.default import train_one_epoch
from visionsuite.engines.classification.val.default import evaluate


import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(x, mean=MEAN, std=STD):
    x *= np.array(std)
    x += np.array(mean)
    x *= 255
    x = x.astype(np.uint8)
    
    return x


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            mkdir(os.path.dirname(cache_path))
            save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            mkdir(os.path.dirname(cache_path))
            save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):

    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    second = now.second
    
    if args.output_dir:
        mkdir(args.output_dir)
        
    args.output_dir = osp.join(args.output_dir, f'{args.model}_{hour}_{minute}_{second}')
    if args.output_dir:
        mkdir(args.output_dir)

    init_distributed_mode(args)
    print(args)

    args.device_ids = parse_device_ids(args.device_ids)
    device = get_device(args.device)
    set_torch_deterministic(args.use_deterministic_algorithms)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

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
    
    model, model_without_ddp = get_model(args.model, device, num_classes, args.distributed, args.sync_bn, args.weights)

    criterion = get_cross_entropy_loss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    optimizer = get_optimizer(args.opt, args.lr, parameters, args.momentum, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = get_scheduler(optimizer, args.lr_scheduler, args.lr_step_size, args.lr_gamma, args.epochs, 
                  args.lr_warmup_method, args.lr_warmup_epochs, args.lr_min, args.lr_warmup_decay)
    model_ema = get_ema_model(model_without_ddp, device, args.model_ema, args.world_size, args.batch_size, args.model_ema_steps, args.model_ema_decay, args.epochs)
    args.start_epoch = set_resume(args.resume, args.ckpt, model_without_ddp, 
                                  optimizer, lr_scheduler, scaler, args.amp)
    

    monitor = Monitor()
    monitor.set(output_dir=args.output_dir, fn='monitor')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_metric_logger = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        
        vis_dir = osp.join(args.output_dir, 'vis')
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)
            
        vis_dir = osp.join(vis_dir, str(epoch))
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)
            
        from _utils.vis.vis_val import save_validation
        save_validation(model, data_loader_test, {0: "1", 1: "2", 2: "3"}, epoch, vis_dir, device, denormalize)
        
        monitor.log({"learning rate": train_metric_logger.meters['lr'].value})
        monitor.log({"train avg loss": train_metric_logger.meters['loss'].avg})
        # for key, val in confmat.values.items():
        #     if 'acc' in key:
        #         monitor.log({key: val})
        #     if 'iou' in key:
        #         monitor.log({key: val})
        monitor.save()
        
        
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
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
