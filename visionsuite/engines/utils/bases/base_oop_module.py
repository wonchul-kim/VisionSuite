import argparse 
from types import SimpleNamespace

class BaseOOPModule:
    def build(self, *args, **kwargs):
        print(f"args: ", args)
        print(f"kwargs: ", kwargs)
        
        if isinstance(kwargs, (argparse.Namespace, SimpleNamespace)):
            self.args = dict(kwargs)
        elif isinstance(kwargs, dict):
            self.args = kwargs
        else:
            NotImplementedError(f"NOT Considered this case for args({args}) and kwargs({kwargs})")
        
        assert self.args is not None, RuntimeError(f"Args for dataset is None")
        
        print(f"Loaded args: {self.args}")
        