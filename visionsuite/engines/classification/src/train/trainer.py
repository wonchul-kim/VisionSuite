from visionsuite.engines.utils.bases.base_oop_module import BaseOOPModule


class Trainer(BaseOOPModule):
    def __init__(self):
        
        self.args = None 
        
        self._loss = None
        self._optimizer = None 
        self._scaler = None 
        self._lr_scheduler = None 
        
    def build(*args, **kwargs):
        super().build(*args, **kwargs)
        
        
        