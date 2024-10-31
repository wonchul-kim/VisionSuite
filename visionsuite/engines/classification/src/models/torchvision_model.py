import torch
import torchvision 

from visionsuite.engines.utils.torch_utils.ema import ExponentialMovingAverage
from visionsuite.engines.utils.registry import MODELS
from visionsuite.engines.utils.helpers import assert_key_dict

        
@MODELS.register()
class TorchvisionModel:
    def __init__(self, **config):
        
        self._config = config
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        self._model = None 
        
        self._load() 
        self.to_device()
        self.set_dist()
        self.set_ema()
        
    def _load(self):
        assert_key_dict(self._config, 'model')
        try:
            self._model = torchvision.models.get_model(name=self.config['model']['model_name'] + self.config['model']['backbone'], 
                                                       num_classes=self.config['model']['num_classes'], 
                                                       weights=self.config['model']['weights'])
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {self.config['model']['model_name']} with config({self.config['model']}): ")
        
    @property
    def config(self):
        return self._config
    
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
        assert_key_dict(self._config, 'device')

        self._device = self._config['device']
        self._model.to(self._device)

    def set_dist(self):
        assert_key_dict(self._config, 'distributed')
        assert_key_dict(self._config, 'sync_bn')

        if self._config['distributed'] and self._config['sync_bn']:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            
        self._model_without_ddp = self._model
        if self._config['distributed']:
            assert_key_dict(self._config, 'gpu')
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self._config['gpu']])
            self._model_without_ddp = self.model.module
            
    def set_ema(self):
        assert_key_dict(self._config, 'ema')
        self._model_ema = None
        if self._config['ema']['use']:
            assert_key_dict(self._config, 'world_size')
            assert_key_dict(self._config, 'batch_size')
            assert_key_dict(self._config, 'epochs')
                
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = self._config['world_size'] * self._config['batch_size'] * self._config['ema_config']['steps'] / self._config['epochs']
            alpha = 1.0 - self._config['ema']['decay']
            alpha = min(1.0, alpha * adjust)
            self._model_ema = ExponentialMovingAverage(self._model_without_ddp, 
                                                 device=self._config['device'], 
                                                 decay=1.0 - alpha)
            
        