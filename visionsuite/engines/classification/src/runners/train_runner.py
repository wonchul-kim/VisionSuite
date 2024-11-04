from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.callbacks import callbacks as cls_callbacks
from visionsuite.engines.classification.src.datasets.build import build_dataset
from visionsuite.engines.classification.src.models.build import build_model
from visionsuite.engines.classification.src.loops.build import build_loop
from visionsuite.engines.classification.utils.registry import RUNNERS
from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner


@RUNNERS.register()
class TrainRunner(BaseTrainRunner):
    def __init__(self):
        super().__init__('classification')
    
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
    def set_variables(self):
        super().set_variables()
        
    def run(self):
        super().run()
                
        dataset = build_dataset(**self.args)
        dataset.build(**self.args['dataset'], distributed=self.args['distributed'])
        
        model = build_model(self.args)
        model.build(**self.args['model'], num_classes=dataset.num_classes, 
                    device=self.args['train']['device'], distributed=self.args['distributed']['use'],
                    sync_bn=self.args['train']['sync_bn'], gpu=self.args['distributed']['gpu'])
        
        callbacks = Callbacks(_callbacks=cls_callbacks)
        loop = build_loop(**self.args['loop'])
        loop.build(_model=model, _dataset=dataset, _archive=self._archive, _callbacks=callbacks, 
                   **self.args)
        loop.run()
        
