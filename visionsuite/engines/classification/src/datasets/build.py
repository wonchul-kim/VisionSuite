from visionsuite.engines.classification.utils.registry import DATASETS

def build_dataset(*args, **kwargs):
    dataset = DATASETS.get(kwargs['dataset']['type'], case_sensitive=True)(transform=kwargs['transform'] if 'transform' in kwargs else None)
    
    return dataset