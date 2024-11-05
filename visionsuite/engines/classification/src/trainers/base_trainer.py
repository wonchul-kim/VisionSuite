import time
import psutil
import torch
import torch.nn as nn

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.metrics.smoothed_value import SmoothedValue
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.classification.utils.registry import TRAINERS
from visionsuite.engines.utils.system.gpu_logger import GPULogger
from visionsuite.engines.utils.bases import BaseOOPModule

@TRAINERS.register()
class BaseTrainer:
    def __init__(self, model, criterion, optimizer, dataloader, device, args, callbacks,
                    model_ema=None, scaler=None, topk=5, archive=None, results=None):
        # super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.callbacks = callbacks
        self.model_ema = model_ema
        self.scaler = scaler
        self.topk = topk
        self.archive = archive
        self.results = results
        
    def run(self, epoch):
        self.model.train()
        gpu_logger = GPULogger(self.args['train']['device_ids'])
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        self.callbacks.run_callbacks('on_train_self._start')
        start_time_epoch = time.time()
        for i, (image, target) in enumerate(metric_logger.log_every(self.dataloader, self.args['train']['print_freq'], header)):
            self.callbacks.run_callbacks('on_train_batch_start')
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(image)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.args['train']['clip_grad_norm'] is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args['train']['clip_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.args['train']['clip_grad_norm'] is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args['train']['clip_grad_norm'])
                self.optimizer.step()

            if self.model_ema and i % self.args['ema']['steps'] == 0:
                self.model_ema.update_parameters(self.model)
                if epoch < self.args['warmup_scheduler']['total_iters']:
                    # Reset ema buffer to keep copying weights during warmup period
                    self.model_ema.n_averaged.fill_(0)

            acc1, acc5 = get_accuracies(output, target, topk=(1, self.topk))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
            gpu_logger.update()
            self.callbacks.run_callbacks('on_train_batch_end')

        self.callbacks.run_callbacks('on_train_epoch_end')

        self.archive.monitor.log({"learning rate": metric_logger.meters['lr'].value})
        self.archive.monitor.log({"train avg loss": metric_logger.meters['loss'].global_avg})
        self.archive.monitor.save()
        
        self.results.epoch = int(epoch)
        self.results.loss = float(round(metric_logger.meters['loss'].global_avg, 4))
        self.results.accuracy = float(round(metric_logger.meters["acc1"].global_avg, 4))
        self.results.learning_rate = float(round(metric_logger.meters['lr'].value, 4))
        self.results.cpu_usage = float(round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 4))
        self.results.gpu_usage = gpu_logger.mean()
        self.results.time_for_a_epoch = float(round(time.time() - start_time_epoch, 3))
        
        gpu_logger.end()

        epoch += 1

@TRAINERS.register()
def base_trainer(model, criterion, optimizer, dataloader, device, epoch, args, callbacks,
                    model_ema=None, scaler=None, topk=5, archive=None, results=None):
    model.train()
    gpu_logger = GPULogger(args['train']['device_ids'])
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    callbacks.run_callbacks('on_train_epoch_start')
    start_time_epoch = time.time()
    for i, (image, target) in enumerate(metric_logger.log_every(dataloader, args['train']['print_freq'], header)):
        callbacks.run_callbacks('on_train_batch_start')
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args['train']['clip_grad_norm'] is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args['train']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args['train']['clip_grad_norm'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args['train']['clip_grad_norm'])
            optimizer.step()

        if model_ema and i % args['ema']['steps'] == 0:
            model_ema.update_parameters(model)
            if epoch < args['warmup_scheduler']['total_iters']:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = get_accuracies(output, target, topk=(1, topk))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        metric_logger.meters['gpu'].update((torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2)
        gpu_logger.update()
        callbacks.run_callbacks('on_train_batch_end')

    callbacks.run_callbacks('on_train_epoch_end')

    archive.monitor.log({"learning rate": metric_logger.meters['lr'].value})
    archive.monitor.log({"train avg loss": metric_logger.meters['loss'].global_avg})
    archive.monitor.save()
    
    results.epoch = int(epoch)
    results.loss = float(round(metric_logger.meters['loss'].global_avg, 4))
    results.accuracy = float(round(metric_logger.meters["acc1"].global_avg, 4))
    results.learning_rate = float(round(metric_logger.meters['lr'].value, 4))
    results.cpu_usage = float(round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 4))
    results.gpu_usage = gpu_logger.mean()
    results.time_for_a_epoch = float(round(time.time() - start_time_epoch, 3))
    
    gpu_logger.end()
