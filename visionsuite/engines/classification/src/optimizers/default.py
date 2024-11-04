from visionsuite.engines.classification.utils.registry import OPTIMIZERS
from visionsuite.engines.utils.torch_utils.utils import set_weight_decay

    
def get_optim_parameters(model, optimizer_config):
    custom_keys_weight_decay = []
    if optimizer_config['bias_weight_decay'] is not None:
        custom_keys_weight_decay.append(("bias", optimizer_config['bias_weight_decay']))
    if optimizer_config['transformer_embedding_decay'] is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, optimizer_config['transformer_embedding_decay']))
            
    parameters = set_weight_decay(
        model,
        optimizer_config['weight_decay'],
        norm_weight_decay=optimizer_config['norm_weight_decay'],
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    return parameters

@OPTIMIZERS.register()
def optimizer(model, optimizer_config):
    
    from visionsuite.engines.utils.helpers import get_params_from_obj
    optim_obj = OPTIMIZERS.get(optimizer_config['type'])
    optim_params = get_params_from_obj(optim_obj)
    for key in optim_params.keys():
        if key in optimizer_config:
            optim_params[key] = optimizer_config[key]
            
        if key == 'params':
            optim_params[key] = get_optim_parameters(model, optimizer_config)
        
    return optim_obj(**optim_params)
    
