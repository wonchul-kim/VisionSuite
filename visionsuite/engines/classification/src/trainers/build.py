from visionsuite.engines.classification.utils.registry import TRAINERS


def build_trainer(**train_config):
    trainer = TRAINERS.get(train_config['type'])
    
    return trainer