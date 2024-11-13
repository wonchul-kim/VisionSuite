from visionsuite.engines.segmentation.utils.registry import RUNNERS
from visionsuite.engines.utils.bases import BaseTestRunner
from visionsuite.engines.utils.callbacks import Callbacks

@RUNNERS.register()
class TestRunner(BaseTestRunner, Callbacks):
    def __init__(self, task):
        BaseTestRunner.__init__(self, task)
        Callbacks.__init__(self)
        
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
        self.run_callbacks('on_runner_set_configs')
        
    def set_variables(self):
        super().set_variables()
        
        self.run_callbacks('on_runner_set_variables')
    
    def run(self):
        super().run()
                
        self.run_callbacks('on_runner_run_start')

        import os.path as osp 
        
        if self.args['augment']['train']['backend'].lower() != "pil" and not self.args['augment']['train']['use_v2']:
            # TODO: Support tensor backend in V1?
            raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
        if self.args['augment']['train']['use_v2'] and self.args['dataset']['type'] != "coco":
            raise ValueError("v2 is only support supported for coco dataset for now.")

        
        self.run_callbacks('on_test_run_end')