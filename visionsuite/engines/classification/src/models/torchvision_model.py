import torch
import torchvision 
import argparse 
from types import SimpleNamespace

from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage
from visionsuite.engines.utils.registry import MODELS
from visionsuite.engines.utils.helpers import assert_key_dict

        
@MODELS.register()
class TorchvisionModel:
    def __init__(self):
        self.args = None
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        self._model = None 
               
    def build(self, *args, **kwargs):
        print(f"args: ", args)
        print(f"kwargs: ", kwargs)
        
        if isinstance(kwargs, (argparse.Namespace, SimpleNamespace)):
            self.args = dict(kwargs)
        elif isinstance(kwargs, dict):
            self.args = kwargs
        else:
            NotImplementedError(f"NOT Considered this case for args({args}) and kwargs({kwargs})")
        
        assert self.args is not None, RuntimeError(f"Args for dataset is None")
        
        print(f"Loaded args: {self.args}")
        
        self.load_model() 
        self.to_device()
        self.set_dist()
        self.set_ema()
        
    def load_model(self):
        try:
            self._model = torchvision.models.get_model(name=self.args['model_name'] + self.args['backbone'], 
                                                       num_classes=self.args['num_classes'], 
                                                       weights=self.args['weights'])
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {self.args['model_name']} with config({self.args}): ")
        
    @property
    def model(self):
        return self._model
    
    @property
    def model_without_ddp(self):
        return self._model_without_ddp
    
    @property
    def model_ema(self):
        return self._model_ema
           
    def to_device(self):
        assert_key_dict(self.args, 'device')
        self._device = self.args['device']
        self._model.to(self._device)

    def set_dist(self):
        assert_key_dict(self.args, 'distributed')
        assert_key_dict(self.args, 'sync_bn')

        if self.args['distributed'] and self.args['sync_bn']:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            
        self._model_without_ddp = self._model
        if self.args['distributed']:
            assert_key_dict(self.args, 'gpu')
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self.args['gpu']])
            self._model_without_ddp = self.model.module
            
    def set_ema(self):
        assert_key_dict(self.args, 'ema')
        self._model_ema = None
        if self.args['ema']['use']:
            assert_key_dict(self.args, 'world_size')
            assert_key_dict(self.args, 'batch_size')
            assert_key_dict(self.args, 'epochs')
                
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = self.args['world_size'] * self.args['batch_size'] * self.args['ema_config']['steps'] / self.args['epochs']
            alpha = 1.0 - self.args['ema']['decay']
            alpha = min(1.0, alpha * adjust)
            self._model_ema = ExponentialMovingAverage(self._model_without_ddp, 
                                                 device=self.args['device'], 
                                                 decay=1.0 - alpha)
            
        