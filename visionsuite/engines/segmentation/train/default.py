import torch
import visionsuite.engines.utils.torch_utils as torch_utils
from visionsuite.engines.utils.smoothed_value import SmoothedValue
from visionsuite.engines.utils.metric_logger import MetricLogger


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target, filename in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image) # 'out': (bs num_classes(including bg) h w)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
    return metric_logger