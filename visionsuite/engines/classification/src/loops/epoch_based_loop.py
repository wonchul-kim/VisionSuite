import os.path as osp

from visionsuite.engines.utils.torch_utils.utils import save_on_master

from visionsuite.engines.classification.src.trainers.default import train_one_epoch
from visionsuite.engines.classification.src.validators.default import val
from visionsuite.engines.classification.utils.registry import LOOPS

@LOOPS.register()
def epoch_based_loop(callbacks, args, train_sampler,
                     model, criterion, optimizer, train_dataloader, scaler, archive,
                     lr_scheduler, val_dataloader, label2class):
        
    callbacks.run_callbacks('on_train_start')
    for epoch in range(args['start_epoch'], args['train']['epochs']):
        if args['distributed']['use']:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model.model, criterion, optimizer, train_dataloader, 
                        args['train']['device'], epoch, args, callbacks, model.model_ema, scaler, 
                        args['train']['topk'], archive)
        lr_scheduler.step()
        
        if archive.weights_dir:
            checkpoint = {
                "model": model.model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model.model_ema:
                checkpoint["model_ema"] = model.model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            save_on_master(checkpoint, osp.join(archive.weights_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, osp.join(archive.weights_dir, "checkpoint.pth"))

        callbacks.run_callbacks('on_val_start')
        val(model.model_ema if model.model_ema else model.model, criterion, val_dataloader, args['train']['device'], epoch, label2class, callbacks, 
                  topk=args['train']['topk'], log_suffix="EMA" if args['model']['ema']['use'] else "", archive=archive)
        callbacks.run_callbacks('on_val_end')
        
    callbacks.run_callbacks('on_train_end')