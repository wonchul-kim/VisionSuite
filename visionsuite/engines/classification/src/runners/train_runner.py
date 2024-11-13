from visionsuite.engines.classification.src.datasets.build import build_dataset
from visionsuite.engines.classification.src.models.build import build_model
from visionsuite.engines.classification.src.loops.build import build_loop
from visionsuite.engines.classification.utils.registry import RUNNERS
from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import callbacks

@RUNNERS.register()
class TrainRunner(BaseTrainRunner, Callbacks):
    def __init__(self, task=""):
        BaseTrainRunner.__init__(self, task)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
    
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
        self.run_callbacks('on_runner_set_configs')
        
    def set_variables(self):
        super().set_variables()
        
        self.run_callbacks('on_runner_set_variables')
               
    def run(self):
        super().run()
                
        self.run_callbacks('on_runner_run_start')

        # TODO: Define transform
        dataset = build_dataset(**self.args['dataset'], transform=None)
        dataset.build(**self.args['dataset'], distributed=self.args['distributed'])
        
        model = build_model(**self.args['model'])
        model.build(**self.args['model'], 
                    num_classes=dataset.num_classes, 
                    train=self.args['train'], 
                    distributed=self.args['distributed']['use']
                )
        
        loop = build_loop(**self.args['loop'])
        loop.build(_model=model, 
                   _dataset=dataset, 
                   _archive=self._archive, 
                   **self.args
                )
        loop.run_loop()
        
        self.run_callbacks('on_runner_run_end')
