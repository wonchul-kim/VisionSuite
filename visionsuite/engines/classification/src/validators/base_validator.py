import torch
import warnings
import os.path as osp 
import os
import time
from abc import abstractmethod

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.functionals import denormalize
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.utils.torch_utils.dist import reduce_across_processes
from visionsuite.engines.classification.utils.registry import VALIDATORS
from visionsuite.engines.utils.bases import BaseOOPModule
from visionsuite.engines.utils.callbacks import Callbacks
from visionsuite.engines.classification.utils.results import ValResults
from .callbacks import callbacks


@VALIDATORS.register()
class BaseValidator(BaseOOPModule, Callbacks):
    def __init__(self):
        BaseOOPModule.__init__(self)
        Callbacks.__init__(self)
        
        self.add_callbacks(callbacks)
        
    def build(self, args, model, criterion, dataloader, device, label2class,
             print_freq=100, log_suffix="", topk=5, archive=None):

        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.label2class = label2class
        self.args = args
        self.print_freq = print_freq
        self.log_suffix = log_suffix
        self.topk = topk
        self.archive = archive
        self.results = ValResults()
        
    @abstractmethod
    def val(self, epoch):
        if epoch%self.args['epoch'] == 0:
            self.model.eval()
            metric_logger = MetricLogger(delimiter="  ")
            header = f"Test: {self.log_suffix}"

            num_processed_samples = 0
            start_time_epoch = 0
            self.run_callbacks('on_val_epoch_start')
            with torch.inference_mode():
                for image, target in metric_logger.log_every(self.dataloader, self.print_freq, header):
                    self.run_callbacks('on_val_batch_start')
                    image = image.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    output = self.model(image)
                    loss = self.criterion(output, target)

                    acc1, acc5 = get_accuracies(output, target, topk=(1, self.topk))
                    # FIXME need to take into account that the datasets
                    # could have been padded in distributed setup
                    batch_size = image.shape[0]
                    metric_logger.update(loss=loss.item())
                    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                    num_processed_samples += batch_size
                    self.run_callbacks('on_val_batch_start')
                    
            # gather the stats from all processes
            num_processed_samples = reduce_across_processes(num_processed_samples)
            if (
                hasattr(self.dataloader.dataset, "__len__")
                and len(self.dataloader.dataset) != num_processed_samples
                and torch.distributed.get_rank() == 0
            ):
                # See FIXME above
                warnings.warn(
                    f"It looks like the dataset has {len(self.dataloader.dataset)} samples, but {num_processed_samples} "
                    "samples were used for the validation, which might bias the results. "
                    "Try adjusting the batch size and / or the world size. "
                    "Setting the world size to 1 is always a safe bet."
                )

            metric_logger.synchronize_between_processes()

            print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
            
            #TODO: MOVE THIS INTO CALLBACK ------------------------------------------------------------
            if self.archive.args['save_val']['use'] and osp.exists(self.archive.val_dir):
                vis_dir = osp.join(self.archive.val_dir, str(epoch))
                if not osp.exists(vis_dir):
                    os.mkdir(vis_dir)
                    
                from visionsuite.engines.classification.utils.vis.vis_val import save_validation
                save_validation(self.model, self.dataloader, self.label2class, epoch, vis_dir, self.device, denormalize)
            # ------------------------------------------------------------------------------------------
                
            self.results.epoch = int(epoch)
            self.results.loss = float(round(metric_logger.meters['loss'].global_avg, 4))
            self.results.accuracy = float(round(metric_logger.meters["acc1"].global_avg, 4))
            self.results.time_for_a_epoch = float(round(time.time() - start_time_epoch, 3))
        
            self.run_callbacks('on_val_epoch_end')


@VALIDATORS.register()
def base_validator(args, model, criterion, dataloader, device, epoch, label2class, callbacks,
             print_freq=100, log_suffix="", topk=5, archive=None, results=None):
    if epoch%args['epoch'] == 0:
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        start_time_epoch = 0
        callbacks.run_callbacks('on_val_epoch_start')
        with torch.inference_mode():
            for image, target in metric_logger.log_every(dataloader, print_freq, header):
                callbacks.run_callbacks('on_val_batch_start')
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc1, acc5 = get_accuracies(output, target, topk=(1, topk))
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
                callbacks.run_callbacks('on_val_batch_start')
                
        # gather the stats from all processes
        callbacks.run_callbacks('on_val_epoch_end')

        num_processed_samples = reduce_across_processes(num_processed_samples)
        if (
            hasattr(dataloader.dataset, "__len__")
            and len(dataloader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(dataloader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes()

        print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
        
        #TODO: MOVE THIS INTO CALLBACK ------------------------------------------------------------
        if archive.args['save_val']['use'] and osp.exists(archive.val_dir):
            vis_dir = osp.join(archive.val_dir, str(epoch))
            if not osp.exists(vis_dir):
                os.mkdir(vis_dir)
                
            from visionsuite.engines.classification.utils.vis.vis_val import save_validation
            save_validation(model, dataloader, label2class, epoch, vis_dir, device, denormalize)
        # ------------------------------------------------------------------------------------------
            
        results.epoch = int(epoch)
        results.loss = float(round(metric_logger.meters['loss'].global_avg, 4))
        results.accuracy = float(round(metric_logger.meters["acc1"].global_avg, 4))
        results.time_for_a_epoch = float(round(time.time() - start_time_epoch, 3))
        