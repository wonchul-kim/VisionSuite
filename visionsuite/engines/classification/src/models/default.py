import torch
from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage
from visionsuite.engines.utils.registry import MODELS


@MODELS.register()
def get_model(model_config):
    
    model_obj = MODELS.get(f"{model_config['backend']}_model" if model_config['backend'] is not None else "")
    model = model_obj(**model_config)
    model.to(model_config['device'])

    if model_config['distributed'] and model_config['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if model_config['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[model_config['gpu']])
        model_without_ddp = model.module

    return model, model_without_ddp

@MODELS.register()
def get_ema_model(model_without_ddp, device, world_size, batch_size, epochs, ema_config):
            
    model_ema = None
    if ema_config['use']:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = world_size * batch_size * ema_config['steps'] / epochs
        alpha = 1.0 - ema_config['decay']
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
        
    return model_ema
        