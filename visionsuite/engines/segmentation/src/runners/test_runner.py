from visionsuite.engines.segmentation.utils.registry import RUNNERS
from visionsuite.engines.utils.bases import BaseTestRunner
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import train_callbacks

@RUNNERS.register()
class TestRunner(BaseTestRunner, Callbacks):
    def __init__(self, task):
        BaseTestRunner.__init__(self, task)
        Callbacks.__init__(self)
        
        self.add_callbacks(train_callbacks)

    def set_configs(self, *args, **kwargs):
        super().set_configs(mode='test', *args, **kwargs)
        
        self.run_callbacks('on_runner_set_configs')
        
    def set_variables(self):
        super().set_variables()
        
        self.run_callbacks('on_runner_set_variables')
    
    def run(self):
        super().run()
                
        self.run_callbacks('on_runner_run_start')
        
        self.run_callbacks('on_test_run_end')