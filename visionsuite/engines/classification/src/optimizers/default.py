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


from visionsuite.engines.utils.torch_utils.utils import set_weight_decay

@OPTIMIZERS.register()
def get_parameters(bias_weight_decay, transformer_embedding_decay, model, weight_decay, norm_weight_decay):
    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, transformer_embedding_decay))
            
    parameters = set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    return parameters