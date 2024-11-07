
def on_train_epoch_start(trainer, *args, **kwargs):
    pass 

def on_train_epoch_end(trainer, *args, **kwargs):
    pass 

def on_train_batch_start(trainer, *args, **kwargs):
    pass

def on_train_batch_end(trainer, *args, **kwargs):
    pass

def on_train_step_start(trainer, *args, **kwargs): # iteration for a batch
    pass

def on_train_step_end(trainer, *args, **kwargs): # iteration for a batch
    pass

callbacks = {
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_train_batch_start": [on_train_batch_start],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_step_start": [on_train_step_start],
    "on_train_step_end": [on_train_step_end],
}

