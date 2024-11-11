import torch
import torchvision 


from visionsuite.engines.segmentation.utils.registry import MODELS
from visionsuite.engines.utils.helpers import assert_key_dict
from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule

        
@MODELS.register()
class TorchvisionModel(BaseOOPModule):
    """
    - weights can be referred by https://pytorch.org/vision/main/models.html
    """
    def __init__(self):
        super().__init__()
        self.args = None
        self._device = 'cpu'
        self._model_without_ddp = None
        self._model_ema = None
        self._model = None 
               
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        
        self.load_model() 
        self.to_device()
        self.set_dist()
        
    @property
    def model(self):
        return self._model
    
    @property
    def model_without_ddp(self):
        return self._model_without_ddp
    
    @property
    def model_ema(self):
        return self._model_ema
    
    @BaseOOPModule.track_status
    def load_model(self):
        try:
            self._model = torchvision.models.get_model(
                                    self.args['model_name'],
                                    weights=self.args['weights'],
                                    weights_backbone=self.args['weights_backbone'],
                                    num_classes=self.args['num_classes'],
                                    aux_loss=self.args['aux_loss'],
                )
            
            print(f"LOADED model: {self.args['model_name']}")
        except Exception as error:
            raise RuntimeError(f"{error}: There has been error when loading torchvision model: {self.args['type']} with config({self.args}): ")
           
    @BaseOOPModule.track_status
    def to_device(self):
        assert_key_dict(self.args['train'], 'device')
        self._device = self.args['train']['device']
        self._model.to(self._device)

    @BaseOOPModule.track_status
    def set_dist(self):
        assert_key_dict(self.args, 'distributed')
        assert_key_dict(self.args['train'], 'sync_bn')

        if self.args['distributed'] and self.args['train']['sync_bn']:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
            
        self._model_without_ddp = self._model
        if self.args['distributed']:
            assert_key_dict(self.args['distributed'], 'gpu')
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self.args['distributed']['gpu']])
            print(f"Distributed Data Parallel is set")
            self._model_without_ddp = self.model.module
        else:
            assert_key_dict(self.args['train'], 'device_ids')
            if len(self.args['train']['device_ids']) > 1:
                self._model = torch.nn.parallel.DataParallel(self._model, device_ids=[self.args['train']['device_ids']])
                self._model_without_ddp = self.model.module
                print(f"Data Parallel is set")
                
                
