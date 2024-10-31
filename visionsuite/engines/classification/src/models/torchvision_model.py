import torch
import torchvision 

from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage
from visionsuite.engines.utils.registry import MODELS

        
@MODELS.register()
class TorchvisionModel:
    def __init__(self, **model_config):
        try:
            self._model = torchvision.models.get_model(name=model_config['model_name'] + model_config['backbone'], 
                                                       num_classes=model_config['num_classes'], 
                                                       weights=model_config['weights'])
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {model_config['model_name']} with config({model_config}): ")
        
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        
    @property
    def model(self):
        return self._model
    
    @property
    def model_without_ddp(self):
        return self._model_without_ddp
    
    @property
    def model_ema(self):
        return self._model_ema
           
    def to_device(self, device):
        self._device = device
        self._model.to(device)

    def set_dist(self, distributed, sync_bn, gpu):

        if distributed and sync_bn:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            
        self._model_without_ddp = self._model
        if distributed:
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[gpu])
            self._model_without_ddp = self.model.module
            
    def set_ema(self, world_size, batch_size, epochs, ema_config):
                
        self._model_ema = None
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
            self._model_ema = ExponentialMovingAverage(self._model_without_ddp, 
                                                 device=self._device, 
                                                 decay=1.0 - alpha)
            
        