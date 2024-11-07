

def on_set_configs(runner, *args, **kwargs):
    pass

def on_set_variables(runner, *args, **kwargs):
    pass

def on_run_start(runner, *args, **kwargs):
    pass

def on_run_end(runner, *args, **kwargs):
    pass

callbacks = {
    "on_set_configs": [on_set_configs],
    "on_set_variables": [on_set_variables],
    "on_run_start": [on_run_start],
    "on_run_end": [on_run_end],
}

