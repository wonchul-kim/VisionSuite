import os.path as osp

from torch.utils.data.dataloader import default_collate

from visionsuite.engines.utils.torch_utils.resume import set_resume
from visionsuite.engines.utils.archives import Archive
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.callbacks import callbacks as cls_callbacks

from visionsuite.engines.classification.utils.augment import get_mixup_cutmix

from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner
from visionsuite.engines.classification.utils.registry import (RUNNERS, MODELS, LOSSES, OPTIMIZERS, 
                                                               SCHEDULERS, LOOPS, PIPELINES, 
                                                               DATASETS, DATALOADERS)

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
        
        train_dataset, val_dataset, self.train_sampler, test_sampler = DATASETS.get("get_datasets")(self.args)
        classes = train_dataset.classes
        print(f"Classes: {classes}")
        self._label2class = {label: _class for label, _class in enumerate(classes)}
        
        num_classes = len(train_dataset.classes)
        self.args.model['num_classes'] = num_classes 
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=self.args.mixup_alpha, cutmix_alpha=self.args.cutmix_alpha, num_classes=num_classes, use_v2=self.args.use_v2
        )
        if mixup_cutmix is not None:

            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        self.train_dataloader = DATALOADERS.get('torch_dataloader')(train_dataset, self.train_sampler, self.args.batch_size, self.args.workers, collate_fn)
        self.val_dataloader = DATALOADERS.get('torch_dataloader')(val_dataset, test_sampler, self.args.batch_size, self.args.workers, collate_fn)
        
    def set_model(self):
        super().set_model()
        self.model = MODELS.get(f"{self.args.model['backend'].capitalize()}Model")(**vars(self.args))
        self.loss = LOSSES.get('get_loss')(self.args.loss)

        parameters = OPTIMIZERS.get('get_parameters')(self.args.bias_weight_decay, self.args.transformer_embedding_decay,
                   self.model.model, self.args.optimizer['weight_decay'],
                   self.args.norm_weight_decay)

        self.optimizer = OPTIMIZERS.get("get_optimizer")(self.args.optimizer, parameters)
        self.scaler = PIPELINES.get('get_scaler')(self.args.amp)
        self.lr_scheduler = SCHEDULERS.get('get_scheduler')(self.optimizer, self.args.epochs, self.args.scheduler)
        self.args.start_epoch = set_resume(self.args.resume, self.args.ckpt, self.model.model_without_ddp, 
                                    self.optimizer, self.lr_scheduler, self.scaler, self.args.amp)

    def run_loop(self):
        super().run_loop()
        
        loop = LOOPS.get('epoch_based_loop')
        loop(self._callbacks, self.args, self.train_sampler, 
                        self.model.model, self.model.model_without_ddp, self.loss, self.optimizer, 
                        self.train_dataloader, self.model.model_ema, self.scaler, self._archive,
                        self.lr_scheduler, self.val_dataloader, self._label2class)
        
