import time
import torch
import torch.nn as nn
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.classification.utils.registry import TRAINERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.results import TrainResults
from .callbacks import callbacks

@TRAINERS.register()
class BaseTrainer(BaseOOPModule, Callbacks):
    def __init__(self):
        BaseOOPModule.__init__(self)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
        
    def build(self, model, criterion, optimizer, lr_scheduler, dataloader, device, args, 
                scaler=None, topk=5, archive=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.scaler = scaler
        self.topk = topk
        self.archive = archive
        self.lr_scheduler = lr_scheduler
        
        self.results = TrainResults()
        self.metric_logger = MetricLogger(delimiter="  ")
        self.gpu_logger = GPULogger(self.args['train']['device_ids'])

    @abstractmethod
    def train(self, epoch):
        self.run_callbacks('on_train_epoch_start')
        self.model.model.train()
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        self.metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        start_time_epoch = time.time()
        for i, (image, target) in enumerate(self.metric_logger.log_every(self.dataloader, self.args['train']['print_freq'], header)):
            self.run_callbacks('on_train_batch_start')
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model.model(image)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.args['train']['clip_grad_norm'] is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.args['train']['clip_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.args['train']['clip_grad_norm'] is not None:
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.args['train']['clip_grad_norm'])
                self.optimizer.step()

            if self.model.model_ema and i % self.args['ema']['steps'] == 0:
                self.model.model_ema.update_parameters(self.model.model)
                if epoch < self.args['warmup_scheduler']['total_iters']:
                    # Reset ema buffer to keep copying weights during warmup period
                    self.model.model_ema.n_averaged.fill_(0)

            acc1, acc5 = get_accuracies(output, target, topk=(1, self.topk))
            batch_size = image.shape[0]
            self.metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            self.metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            self.metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            self.metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            self.metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
            self.gpu_logger.update()

            self.run_callbacks('on_train_batch_end')

        self.lr_scheduler.step()

        self.run_callbacks('on_train_epoch_end', 
                           epoch=epoch, start_time_epoch=start_time_epoch)

