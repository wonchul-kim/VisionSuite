import torch 

def set_resume(resume, ckpt, model_without_ddp, 
               optimizer, lr_scheduler, scaler, amp, test_only=False):
    if resume:
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not test_only)
        if not test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            if amp:
                scaler.load_state_dict(checkpoint["scaler"])
                
    return start_epoch