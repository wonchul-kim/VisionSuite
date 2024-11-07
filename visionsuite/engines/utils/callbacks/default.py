from collections import defaultdict
from copy import deepcopy

### TrainRunner -------------------------------------
def on_set_configs(runner, *args, **kwargs):
    pass

def on_set_variables(runner, *args, **kwargs):
    pass

def on_run_start(runner, *args, **kwargs):
    pass

def on_run_end(runner, *args, **kwargs):
    pass

### Loop ---------------------------------------------
def on_build_start(loop, *args, **kwargs):
    pass

def on_build_end(loop, *args, **kwargs):
    pass

def on_run_loop_start(loop, *args, **kwargs):
    pass

def on_run_loop_end(loop, *args, **kwargs):
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



default_callbacks = {
    ### TrainRunner -------------------------------
    "on_set_configs": [on_set_configs],
    "on_set_variables": [on_set_variables],
    "on_run_start": [on_run_start],
    "on_run_end": [on_run_end],
    
    ### Loop -------------------------------
    "on_build_start": [on_build_start],
    "on_build_end": [on_build_end],
    "on_run_loop_start": [on_run_loop_start],
    "on_run_loop_end": [on_run_loop_end],
    
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


def get_default_callbacks():
    return defaultdict(list, deepcopy(default_callbacks))
