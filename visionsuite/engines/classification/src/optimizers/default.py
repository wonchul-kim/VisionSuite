import torch
from visionsuite.engines.classification.utils.registry import OPTIMIZERS

@OPTIMIZERS.register()
def get_optimizer(optimizer_config, parameters):

    if optimizer_config['optimizer_name'].startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
            weight_decay=optimizer_config['weight_decay'],
            nesterov="nesterov" in optimizer_config['optimizer_name'],
        )
    elif optimizer_config['optimizer_name'] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=optimizer_config['lr'], 
            momentum=optimizer_config['momentum'], 
            weight_decay=optimizer_config['weight_decay'], 
            eps=0.0316, alpha=0.9
        )
    elif optimizer_config['optimizer_name'] == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
    else:
        raise RuntimeError(f"Invalid optimizer {optimizer_config['optimizer_name']}. Only SGD, RMSprop and AdamW are supported.")
    
    
    return optimizer