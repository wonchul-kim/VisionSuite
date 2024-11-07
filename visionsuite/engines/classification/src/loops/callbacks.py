def on_build_start(loop, *args, **kwargs):
    pass

def on_build_end(loop, *args, **kwargs):
    
    assert loop is not None, ValueError(f"loop is None")
    
    for attribute_name in loop.required_attributes:
        assert hasattr(loop, attribute_name), ValueError(f'{attribute_name} must be assgined in loop class')
        
    assert loop.current_epoch is not None, ValueError(f"Current epoch is None")
    

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
