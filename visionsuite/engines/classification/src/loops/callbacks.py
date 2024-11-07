def on_build_start(loop, *args, **kwargs):
    pass

def on_build_end(loop, *args, **kwargs):
    pass

def on_run_loop_start(loop, *args, **kwargs):
    pass

def on_run_loop_end(loop, *args, **kwargs):
    pass

callbacks = {
    "on_build_start": [on_build_start],
    "on_build_end": [on_build_end],
    "on_run_loop_start": [on_run_loop_start],
    "on_run_loop_end": [on_run_loop_end],
}
