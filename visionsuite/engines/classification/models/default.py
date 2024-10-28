import torchvision
import torch
from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage

def get_model(model, device, num_classes, distributed, sync_bn, weights, gpu):
    print("Creating model")
    model = torchvision.models.get_model(model, weights=weights, num_classes=num_classes)
    model.to(device)

    if distributed and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_without_ddp = model.module

    return model, model_without_ddp

def get_ema_model(model_without_ddp, device, model_ema, world_size, batch_size, model_ema_steps, model_ema_decay, epochs):
            
    model_ema = None
    if model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = world_size * batch_size * model_ema_steps / epochs
        alpha = 1.0 - model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
        
    return model_ema
        