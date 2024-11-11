# import os

# import torch
# from visionsuite.engines.segmentation.train.default import train_one_epoch
# from visionsuite.engines.segmentation.val.default import evaluate
# from visionsuite.engines.segmentation.optimizers.default import get_optimizer
# from visionsuite.engines.segmentation.schedulers.default import get_scheduler
# from visionsuite.engines.segmentation.models.default import get_model
# from visionsuite.engines.segmentation.src.datasets.build import get_dataset
# from visionsuite.engines.segmentation.src.dataloaders.build import get_dataloader
# from visionsuite.engines.segmentation.losses.default import criterion
# from visionsuite.engines.utils.torch_utils.utils import save_on_master
# from visionsuite.engines.utils.loggers.monitor import Monitor
from visionsuite.engines.segmentation.src.datasets.build import build_dataset
from visionsuite.engines.classification.src.models.build import build_model
from visionsuite.engines.segmentation.utils.registry import RUNNERS
from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner
from visionsuite.engines.utils.callbacks import Callbacks

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
    def __init__(self, task=""):
        BaseTrainRunner.__init__(self, task)
        Callbacks.__init__(self)
        
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
        self.run_callbacks('on_set_configs')
        
    def set_variables(self):
        super().set_variables()
        
        self.run_callbacks('on_set_variables')
               
    def run(self):
        super().run()
                
        self.run_callbacks('on_run_start')

        import os.path as osp 
        
        if self.args['augment']['train']['backend'].lower() != "pil" and not self.args['augment']['train']['use_v2']:
            # TODO: Support tensor backend in V1?
            raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
        if self.args['augment']['train']['use_v2'] and self.args['dataset']['type'] != "coco":
            raise ValueError("v2 is only support supported for coco dataset for now.")



        # TODO: Define transform
        dataset = build_dataset(**self.args['dataset'], transform=None)
        dataset.build(**self.args['dataset'], distributed=self.args['distributed'])
        
        model = build_model(**self.args['model'])
        model.build(**self.args['model'], 
                    num_classes=dataset.num_classes, 
                    train=self.args['train'], 
                    distributed=self.args['distributed']['use']
        
        # dataset, num_classes = get_dataset(self.args, is_train=True)
        # dataset_test, _ = get_dataset(self.args, is_train=False)
        # data_loader, data_loader_test, train_sampler, test_sampler = get_dataloader(self.args, dataset, dataset_test)
        # model, model_without_ddp, params_to_optimize = get_model(self.args, num_classes, self.args['device'])
        
        # optimizer = get_optimizer(self.args, params_to_optimize)
        # scaler = torch.cuda.amp.GradScaler() if self.args['amp'] else None
        # iters_per_epoch = len(data_loader)
        # lr_scheduler = get_scheduler(self.args, optimizer, iters_per_epoch)

        # # self.args['start_epoch'] = set_resume(self.args['resume'], self.args['ckpt'], model_without_ddp, 
        # #                               optimizer, lr_scheduler, scaler, self.args['amp'])
        
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

        #     checkpoint = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_scheduler": lr_scheduler.state_dict(),
        #         "epoch": epoch,
        #         "self.args": self.args,
        #         'scale': scaler.state_dict() if scaler and self.args['amp'] else None
        #     }
        #     save_on_master(checkpoint, os.path.join(self.args['output_dir'], f"model_{epoch}.pth"))
        #     save_on_master(checkpoint, os.path.join(self.args['output_dir'], "checkpoint.pth"))

