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
    hour = now.hour
    minute = now.minute
    second = now.second
    
    output_dir = osp.join(output_dir, f'{year}_{month}_{hour}_{minute}_{second}')
    mkdir(output_dir, make_dirs=make_dirs)
    
    return output_dir
    
def yaml2namespace(args_file):
    import argparse
    import yaml 

    
    with open(args_file, 'r') as yf:
        cfgs = yaml.safe_load(yf)
        
    args = argparse.Namespace(**cfgs)
    
    return args
