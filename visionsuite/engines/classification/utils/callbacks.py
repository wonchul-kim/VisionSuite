

def on_set_configs(*args, **kwargs):
    pass

def on_set_variables(*args, **kwargs):
    pass

def on_set_dataset(*args, **kwargs):
    pass

### Model -------------------------------------
def on_set_model(*args, **kwargs):
    pass

### Train -------------------------------------
def on_train_start(*args, **kwargs):
    pass

def on_train_end(*args, **kwargs):
    pass

def on_train_epoch_start(*args, **kwargs):
    pass

def on_train_epoch_end(*args, **kwargs):
    pass

def on_train_batch_start(*args, **kwargs):
    pass

def on_train_batch_end(*args, **kwargs):
    pass

def on_train_step_start(*args, **kwargs): # iteration for a batch
    pass

def on_train_step_end(*args, **kwargs): # iteration for a batch
    pass

### Val. -------------------------------------
def on_val_start(*args, **kwargs):
    pass

def on_val_end(*args, **kwargs):
    pass

def on_val_epoch_start(*args, **kwargs):
    pass

def on_val_epoch_end(*args, **kwargs):
    pass

def on_val_batch_start(*args, **kwargs):
    pass

def on_val_batch_end(*args, **kwargs):
    pass

def on_val_step_start(*args, **kwargs): # iteration for a batch
    pass

def on_val_step_end(*args, **kwargs): # iteration for a batch
    pass

### End -------------------------------------
def on_end_start(*args, **kwargs):
    pass

def on_end_end(*args, **kwargs):
    pass



callbacks = {
    "on_set_configs": [on_set_configs],
    "on_set_variables": [on_set_variables],
    "on_set_dataset": [on_set_dataset],
    
    "on_set_model": [on_set_model],
    
    "on_train_start": [on_train_start],
    "on_train_end": [on_train_end],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_train_batch_start": [on_train_epoch_start],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_step_start": [on_train_step_start],
    "on_train_step_end": [on_train_step_end],

    "on_val_start": [on_val_start],
    "on_val_end": [on_val_end],
    "on_val_epoch_start": [on_val_epoch_start],
    "on_val_epoch_end": [on_val_epoch_end],
    "on_val_batch_start": [on_val_epoch_start],
    "on_val_batch_end": [on_val_epoch_end],
    "on_val_step_start": [on_val_step_start],
    "on_val_step_end": [on_val_step_end],
   
    "on_end_start": [on_end_start],
    "on_end_end": [on_end_end],
}
