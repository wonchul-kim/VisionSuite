
def on_val_epoch_start(validator, *args, **kwargs):
    pass

def on_val_epoch_end(validator, *args, **kwargs):
    pass

def on_val_batch_start(validator, *args, **kwargs):
    pass

def on_val_batch_end(validator, *args, **kwargs):
    pass

def on_val_step_start(validator, *args, **kwargs): # iteration for a batch
    pass

def on_val_step_end(validator, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_val_epoch_start": [on_val_epoch_start],
    "on_val_epoch_end": [on_val_epoch_end],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_step_start": [on_val_step_start],
    "on_val_step_end": [on_val_step_end],
}
