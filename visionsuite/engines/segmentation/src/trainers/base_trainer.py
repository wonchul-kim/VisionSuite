import time
import torch
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.segmentation.utils.registry import TRAINERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.segmentation.utils.results import TrainResults
# from .callbacks import callbacks

@TRAINERS.register()
class BaseTrainer(BaseOOPModule, Callbacks):
    
    required_attributes = ['model', 'loss', 'dataloader']
    
    def __init__(self):
        BaseOOPModule.__init__(self)
        Callbacks.__init__(self)
        
        # self.add_callbacks(callbacks)
        
    def build(self, model, loss, optimizer, lr_scheduler, dataloader, scaler=None, archive=None, **args):
        
        self.run_callbacks('on_build_trainer_start')
        
        self.epoch = None
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.args = args
        self.scaler = scaler
        self.archive = archive
        
        self.results = TrainResults()
        self.metric_logger = MetricLogger(delimiter="  ")
        self.gpu_logger = GPULogger(self.args['device_ids'])

        
        self.run_callbacks('on_build_trainer_end')

    # @abstractmethod
    # def train(self, epoch):
    #     self.run_callbacks('on_train_epoch_start')
    #     self.epoch = epoch
    #     self.model.model.train()
    #     self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    #     self.metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    #     header = f"Epoch: [{epoch}]"
    #     start_time_epoch = time.time()
    #     for step, (image, target) in enumerate(self.metric_logger.log_every(self.dataloader, self.args['print_freq'], header)):
    #         self.run_callbacks('on_train_batch_start')
    #         start_time = time.time()
    #         image, target = image.to(self.args['device']), target.to(self.args['device'])
    #         with torch.cuda.amp.autocast(enabled=self.scaler is not None):
    #             output = self.model.model(image)
    #             loss = self.loss(output, target)

    #         self._backward(loss)
    #         self._reset_ema_buffer(epoch, step)
    #         self._update_logger(output, target, loss, start_time, batch_size = image.shape[0])

    #         self.run_callbacks('on_train_batch_end')

    #     self.lr_scheduler.step()

    #     self.run_callbacks('on_train_epoch_end', 
    #                        epoch=epoch, start_time_epoch=start_time_epoch)

    @abstractmethod
    def train(self, epoch):
        start_time_epoch = time.time()
        self.model.model.train()
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch: [{epoch}]"
        for batch in self.metric_logger.log_every(self.dataloader, self.args['print_freq'], header):
            start_time = time.time()
            image, target = batch[0].to(self.args['device']), batch[1].to(self.args['device'])
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model.model(image) # 'out': (bs num_classes(including bg) h w)
                loss = self.loss(output, target)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            self._update_logger(output, target, loss, start_time, batch_size = image.shape[0])

                
    def _update_logger(self, output, target, loss, start_time, batch_size):
        if self.metric_logger is not None:
            self.metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            # acc1, acc5 = get_accuracies(output, target, topk=(1, self.args['topk']))
            # self.metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            # self.metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            # self.metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            # self.metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            # self.metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
        
        if self.gpu_logger is not None:
            self.gpu_logger.update()
        