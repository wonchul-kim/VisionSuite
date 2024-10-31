import time
import torch
import torch.nn as nn
from visionsuite.engines.utils.metric_logger import MetricLogger
from visionsuite.engines.utils.smoothed_value import SmoothedValue
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, args, callbacks,
                    model_ema=None, scaler=None, topk=5, archive=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    callbacks.run_callbacks('on_train_epoch_start')
    for i, (image, target) in enumerate(metric_logger.log_every(dataloader, args.print_freq, header)):
        callbacks.run_callbacks('on_train_batch_start')
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = get_accuracies(output, target, topk=(1, topk))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        callbacks.run_callbacks('on_train_batch_end')

    callbacks.run_callbacks('on_train_epoch_end')

        
    archive.monitor.log({"learning rate": metric_logger.meters['lr'].value})
    archive.monitor.log({"train avg loss": metric_logger.meters['loss'].avg})
    # for key, val in confmat.values.items():
    #     if 'acc' in key:
    #         archive.monitor.log({key: val})
    #     if 'iou' in key:
    #         archive.monitor.log({key: val})
    archive.monitor.save()

    return metric_logger
