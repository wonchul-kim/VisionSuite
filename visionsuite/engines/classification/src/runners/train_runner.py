import os.path as osp

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.utils.archives import Archive
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.callbacks import callbacks as cls_callbacks

from visionsuite.engines.classification.utils.augment import get_mixup_cutmix
from visionsuite.engines.classification.src.dataloaders.default import get_dataloader

from visionsuite.engines.classification.src.datasets.directory_dataset import get_datasets

from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner
from visionsuite.engines.classification.utils.registry import (RUNNERS, MODELS, LOSSES, OPTIMIZERS, 
                                                               SCHEDULERS, LOOPS, PIPELINES)

@RUNNERS.register()
class TrainRunner(BaseTrainRunner):
    def __init__(self):
        super().__init__('classification')
    
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
    def set_variables(self):
        super().set_variables()

        self._archive = Archive(osp.join(self.args.output_dir, 'classification'), monitor=True)
        self._archive.save_args(self.args)
        
        self._callbacks = Callbacks(_callbacks=cls_callbacks)
        
    def set_dataset(self):
        super().set_dataset()
        
        dataset, dataset_test, self.train_sampler, test_sampler = get_datasets(self.args)
        classes = dataset.classes
        print(f"Classes: {classes}")
        self._label2class = {label: _class for label, _class in enumerate(classes)}
        
        num_classes = len(dataset.classes)
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=self.args.mixup_alpha, cutmix_alpha=self.args.cutmix_alpha, num_classes=num_classes, use_v2=self.args.use_v2
        )
        if mixup_cutmix is not None:

            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        self.data_loader, self.data_loader_test = get_dataloader(dataset, dataset_test, self.train_sampler, test_sampler, self.args.batch_size, self.args.workers, collate_fn)
        
        
        self.args.model['num_classes'] = num_classes 
        self.args.model['distributed'] = self.args.distributed
        self.args.model['device'] = self.args.device
        self.args.model['sync_bn'] = self.args.sync_bn
        self.args.model['gpu'] = self.args.gpu
        self.model, self.model_without_ddp = MODELS.get("get_model")(self.args.model)
        self.model_ema = MODELS.get('get_ema_model')(self.model_without_ddp, self.args.device, 
                                       self.args.world_size, self.args.batch_size, self.args.epochs,
                                       self.args.ema)
        
        self._loss = LOSSES.get('get_loss')(self.args.loss)

        parameters = OPTIMIZERS.get('get_parameters')(self.args.bias_weight_decay, self.args.transformer_embedding_decay,
                   self.model, self.args.optimizer['weight_decay'],
                   self.args.norm_weight_decay)

        self._optimizer = OPTIMIZERS.get("get_optimizer")(self.args.optimizer, parameters)
        self._scaler = PIPELINES.get('get_scaler')(self.args.amp)
        self._lr_scheduler = SCHEDULERS.get('get_scheduler')(self._optimizer, self.args.epochs, self.args.scheduler)
        self.args.start_epoch = set_resume(self.args.resume, self.args.ckpt, self.model_without_ddp, 
                                    self._optimizer, self._lr_scheduler, self._scaler, self.args.amp)

    def run_loop(self):
        super().run_loop()
        
        loop = LOOPS.get('epoch_based_loop')
        loop(self._callbacks, self.args, self.train_sampler, 
                        self.model, self.model_without_ddp, self._loss, self._optimizer, 
                        self.data_loader, self.model_ema, self._scaler, self._archive,
                        self._lr_scheduler, self.data_loader_test, self._label2class)
        
