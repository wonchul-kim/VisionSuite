from visionsuite.engines.segmentation.src.loops.build import build_loop
from visionsuite.engines.segmentation.src.datasets.build import build_dataset
from visionsuite.engines.segmentation.src.models.build import build_model
from visionsuite.engines.segmentation.utils.registry import RUNNERS
from visionsuite.engines.utils.bases import BaseTrainRunner
from visionsuite.engines.utils.callbacks import Callbacks
from .callbacks import callbacks

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(x, mean=MEAN, std=STD):
    x *= np.array(std)
    x += np.array(mean)
    x = x.astype(np.uint8)
    
    return x

@RUNNERS.register()
class TrainRunner(BaseTrainRunner, Callbacks):
    def __init__(self, task="", name="TrainRunner"):
        BaseTrainRunner.__init__(self, task=task, name=name)
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
        
        if self.args['augment']['train']['backend'].lower() != "pil" and not self.args['augment']['train']['use_v2']:
            # TODO: Support tensor backend in V1?
            raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
        if self.args['augment']['train']['use_v2'] and self.args['dataset']['type'] != "coco":
            raise ValueError("v2 is only support supported for coco dataset for now.")

        # TODO: Define transform
        dataset = build_dataset(**self.args['dataset'], transform=None)
        dataset.build(**self.args['dataset'], distributed=self.args['distributed'], _logger=self.args['dataset'].get('logger', None))
        self.log_info(f"Dataset is LOADED and BUILT", self.run.__name__, __class__.__name__)
        
        model = build_model(**self.args['model'])
        model.build(**self.args['model'], 
                    num_classes=dataset.num_classes, 
                    train=self.args['train'], 
                    distributed=self.args['distributed']['use'],
                    _logger=self.args['model'].get('logger', None)
        )
        self.log_info(f"Model is LOADED and BUILT", self.run.__name__, __class__.__name__)
        
        loop = build_loop(**self.args['loop'])
        loop.build(_model=model, 
                   _dataset=dataset, 
                   _archive=self._archive, 
                   **self.args,
                   _logger=self.args['loop'].get('logger', None)
                )
        self.log_info(f"Loop is LOADED and BUILT", self.run.__name__, __class__.__name__)

        loop.run_loop()
        
        self.run_callbacks('on_runner_run_end')

        
        # self.args['start_epoch'] = 1
        # monitor = Monitor()
        # monitor.set(output_dir=self.args['output_dir'], fn='monitor')
        # for epoch in range(self.args['start_epoch'], self.args['epochs']):
        #     if self.args['distributed']['use']:
        #         train_sampler.set_epoch(epoch)
        #     train_metric_logger = train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, self.args['device'], epoch, self.args['print_freq'], scaler)
        #     confmat, val_metric_logger = evaluate(model, data_loader_test, device=self.args['device'], num_classes=num_classes)
            
        #     from utils.vis.vis_val import save_validation
        #     vis_dir = osp.join(self.args['output_dir'], f'vis/{epoch}')
        #     if not osp.exists(vis_dir):
        #         os.makedirs(vis_dir)
            
        #     save_validation(model, self.args['device'], dataset_test, 4, epoch, vis_dir, denormalize)
        #     print(confmat)
            
        #     monitor.log({"learning rate": train_metric_logger.meters['lr'].value})
        #     monitor.log({"train avg loss": train_metric_logger.meters['loss'].avg})
        #     for key, val in confmat.values.items():
        #         if 'acc' in key:
        #             monitor.log({key: val})
        #         if 'iou' in key:
        #             monitor.log({key: val})
        #     monitor.save()
