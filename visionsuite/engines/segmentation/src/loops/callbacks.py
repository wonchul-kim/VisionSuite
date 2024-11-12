def on_build_loop_start(loop, *args, **kwargs):
    pass

def on_build_loop_end(loop, *args, **kwargs):
    
    assert loop is not None, ValueError(f"loop is None")
    
    for attribute_name in loop.required_attributes:
        assert hasattr(loop, attribute_name), ValueError(f'{attribute_name} must be assgined in loop class')
        assert getattr(loop, attribute_name) is not None, ValueError(f"{attribute_name} is None for loop")
    

def on_run_loop_start(loop, *args, **kwargs):
    pass

def on_run_loop_end(loop, *args, **kwargs):
    pass

callbacks = {
    "on_build_loop_start": [on_build_loop_start],
    "on_build_loop_end": [on_build_loop_end],
    "on_run_loop_start": [on_run_loop_start],
    "on_run_loop_end": [on_run_loop_end],
}
