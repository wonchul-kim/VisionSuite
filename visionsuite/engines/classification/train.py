import datetime
import os
import time
import warnings
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


import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(x, mean=MEAN, std=STD):
    x *= np.array(std)
    x += np.array(mean)
    x *= 255
    x = x.astype(np.uint8)
    
    return x

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 3))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return metric_logger

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 3))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


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
    cache_path = _get_cache_path(traindir)
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
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
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
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

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
        utils.mkdir(args.output_dir)
        
    args.output_dir = osp.join(args.output_dir, f'{args.model}_{hour}_{minute}_{second}')
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

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

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

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
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))



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
