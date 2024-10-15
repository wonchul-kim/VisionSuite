import datetime
import os
import time

import torch
import torch.utils.data
import utils
from datetime import datetime
from visionsuite.engines.segmentation.train.default import train_one_epoch
from visionsuite.engines.segmentation.val.default import evaluate
from visionsuite.engines.segmentation.optimizers.default import get_optimizer
from visionsuite.engines.segmentation.schedulers.default import get_scheduler
from visionsuite.engines.segmentation.models.default import get_model
from visionsuite.engines.segmentation.datasets.default import get_dataset
from visionsuite.engines.segmentation.dataloaders.default import get_dataloader
from visionsuite.engines.segmentation.losses.default import criterion



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

    args.device_ids = list(map(int, args.device_ids.split(",")))

    now = datetime.now()
    hour = now.hour
    minute = now.minute
    second = now.second
    
    args.output_dir += f'_{args.model}_{hour}_{minute}_{second}'
    
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

    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)

    data_loader, data_loader_test, train_sampler, test_sampler = get_dataloader(args, dataset, dataset_test)

    model, model_without_ddp, params_to_optimize = get_model(args, num_classes, device)
    
    optimizer = get_optimizer(args, params_to_optimize)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    lr_scheduler = get_scheduler(args, optimizer, iters_per_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    lrs, losses, acces, ious = [], [], {}, {}
    import matplotlib.pyplot as plt 
    import seaborn as sns
    sns.set_style('darkgrid')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_metric_logger = train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat, val_metric_logger = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        
        from vis.vis_val import save_validation
        vis_dir = osp.join(args.output_dir, f'vis/{epoch}')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        
        save_validation(model, device, dataset_test, 4, epoch, vis_dir, denormalize)
        print(confmat)
        
        lrs.append(train_metric_logger.meters['lr'].value)
        losses.append(train_metric_logger.meters['loss'].avg)
        for key, val in confmat.values.items():
            if 'acc' in key:
                if key not in acces:
                    acces[key] = []
                acces[key].append(val)
            if 'iou' in key:
                if key not in ious:
                    ious[key] = []
                ious[key].append(val)

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        logs_dir = osp.join(args.output_dir, 'logs')
        if not osp.exists(logs_dir):
            os.makedirs(logs_dir)
        fig = plt.figure()
        plt.plot(lrs)
        fig.savefig(osp.join(logs_dir, 'lrs.jpg'))
        plt.close()
        fig = plt.figure()
        plt.plot(losses)
        fig.savefig(osp.join(logs_dir, 'losses.jpg'))
        plt.close()
        fig = plt.figure()
        for key, val in acces.items():
            plt.plot(val, label=key)
        plt.legend()
        fig.savefig(osp.join(logs_dir, 'acces.jpg'))
        plt.close()
        fig = plt.figure()
        for key, val in ious.items():
            plt.plot(val, label=key)
        plt.legend()
        fig.savefig(osp.join(logs_dir, 'ious.jpg'))
        plt.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--output-dir", default="/HDD/_projects/benchmark/semantic_segmentation/new_model/outputs/torch/dlv3_scratch_tear_stabbed", type=str)
    parser.add_argument("--weights-file", default="/HDD/_projects/benchmark/semantic_segmentation/new_model/outputs/torch/dlv3_scratch_tear_stabbed_deeplabv3_resnet50_20_22_21/model_99.pth", type=str)
    # parser.add_argument("--data-path", default="/HDD/datasets/public/coco", type=str, help="dataset path")

    # parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--data-path", default="/HDD/_projects/benchmark/semantic_segmentation/new_model/datasets/split_datasets/scratch_tear_stabbed", type=str, help="dataset path")
    parser.add_argument("--dataset", default="mask", type=str, help="dataset name")
    parser.add_argument("--model", default="deeplabv3_resnet50", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device-ids", default="0", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default='ResNet50_Weights.IMAGENET1K_V1', type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
