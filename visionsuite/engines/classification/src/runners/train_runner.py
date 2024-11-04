import torch
from torch.utils.data.dataloader import default_collate

from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.callbacks import callbacks as cls_callbacks

from visionsuite.engines.classification.utils.augment import get_mixup_cutmix

from visionsuite.engines.utils.bases.base_train_runner import BaseTrainRunner
from visionsuite.engines.classification.src.datasets.build import build_dataset
from visionsuite.engines.classification.src.models.build import build_model
from visionsuite.engines.classification.src.loops.build import build_loop
from visionsuite.engines.classification.utils.registry import RUNNERS
                                                               


@RUNNERS.register()
class TrainRunner(BaseTrainRunner):
    def __init__(self):
        super().__init__('classification')
    
    def set_configs(self, *args, **kwargs):
        super().set_configs(*args, **kwargs)
        
    def set_variables(self):
        super().set_variables()
        
    def run(self):
        super().start_train()
        
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
            
        import torchvision.transforms as transforms
        transform = transforms.Compose(
                                        [transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
        
        dataset = build_dataset(**self.args, transform=transform)
        dataset.build(**self.args['dataset'], distributed=self.args['distributed'])
        
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=self.args['mixup_alpha'], cutmix_alpha=self.args['cutmix_alpha'], num_classes=len(dataset.train_dataset.classes) , use_v2=self.args['use_v2']
        )
        if mixup_cutmix is not None:

            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))
        else:
            collate_fn = default_collate

        model = build_model(self.args)
        model.build(**self.args['model'], num_classes=dataset.num_classes, 
                    device=self.args['device'], distributed=self.args['distributed'],
                    sync_bn=self.args['sync_bn'], gpu=self.args['gpu'])
        
        callbacks = Callbacks(_callbacks=cls_callbacks)
        loop = build_loop()
        loop.build(**self.args)
        loop.run(model, dataset, self._archive, callbacks, collate_fn)
        
