import os 
import warnings
import datetime
import os.path as osp

def mkdir(path, make_dirs=False):
    if not osp.exists(path):
        if make_dirs:
            os.makedirs(path)
        else:
            os.mkdir(path)
            
        print(f"Created directory: {path}")
    else:
        warnings.warn(f"There is already {path}")

def get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def create_output_dir(output_dir, make_dirs=False):
    now = datetime.datetime.now()
    year = now.year 
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    
    output_dir = osp.join(output_dir, f'{year}_{month}_{day}_{hour}_{minute}_{second}')
    mkdir(output_dir, make_dirs=make_dirs)
    
    return output_dir
    
def yaml2namespace(args_file):
    import argparse
    import yaml 

    
    with open(args_file, 'r') as yf:
        cfgs = yaml.safe_load(yf)
        
    args = argparse.Namespace(**cfgs)
    
    return args

def assert_key_dict(dictionary, key):
    assert key in dictionary, ValueError(f"There is no key({key})")
    
def get_params_from_obj(obj):
    import inspect

    signature = inspect.signature(obj)

    # Extract the parameters
    parameters = {param.name: param.default for param in signature.parameters.values()}
    
    return parameters
