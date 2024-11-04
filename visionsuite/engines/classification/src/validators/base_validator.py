import torch
import warnings
import os.path as osp 
import os

from visionsuite.engines.utils.metrics.metric_logger import MetricLogger
from visionsuite.engines.utils.functionals import denormalize
from visionsuite.engines.classification.utils.metrics.accuracy import get_accuracies
from visionsuite.engines.utils.torch_utils.dist import reduce_across_processes
from visionsuite.engines.classification.utils.registry import VALIDATORS


@VALIDATORS.register()
def base_validator(model, criterion, dataloader, device, epoch, label2class, callbacks,
             print_freq=100, log_suffix="", topk=5, archive=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
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
    
    if archive:
        vis_dir = osp.join(archive.val_dir, str(epoch))
        if not osp.exists(vis_dir):
            os.mkdir(vis_dir)
            
        from visionsuite.engines.classification.utils.vis.vis_val import save_validation
        save_validation(model, dataloader, label2class, epoch, vis_dir, device, denormalize)
    
    return metric_logger.acc1.global_avg