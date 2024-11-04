from visionsuite.engines.classification.utils.registry import VALIDATORS

def build_validator(**val_config):
    validator = VALIDATORS.get(val_config['type'])
    
    return validator